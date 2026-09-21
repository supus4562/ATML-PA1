"""task3/models/backbone.py — ResNet-18 feature extractor for Task 3."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """ResNet-18 with fc removed, returns 512-d pooled features.

    Includes freeze_bn() method to apply BN freeze policy (required by all methods).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        # Remove the final fully connected layer; keep everything up to avgpool
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = 512

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # (B, 512)

    def freeze_bn(self) -> None:
        """Set all BatchNorm2d layers to eval mode.

        Freezes running mean/var while keeping gamma/beta trainable.
        Must be called after model.train() in every training step.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
