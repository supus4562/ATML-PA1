import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, dtype=torch.float32, device=x.device))
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return grad_output.neg() * alpha, None

class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha):
        x = self.grl(x, alpha)
        return self.net(x)
