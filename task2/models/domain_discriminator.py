"""task2/models/domain_discriminator.py — Domain discriminator with Gradient Reversal Layer.

Architecture follows the PA spec exactly:
  "a 256-unit hidden layer, ReLU, dropout of 0.5, and a two-class output layer"
  attached to the 512-d feature (or 3584-d for CDAN).
"""
import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward: identity.
    Backward: multiplies incoming gradient by -alpha.
    This causes the feature extractor to maximise the domain loss
    while the discriminator head minimises it — standard GRL minimax.
    """
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
    """Binary domain classifier (source=0, target=1).

    Architecture as specified by the PA:
        Linear(in_dim, 256) → ReLU() → Dropout(0.5) → Linear(256, 2)

    NOTE: No normalisation layers are added. The PA spec explicitly defines
    this architecture. Stability under unit-weight adversarial training is
    achieved via:
      - Per-player independent gradient clipping (max_norm=1.0)
      - Separate AdamW optimizers for discriminator vs backbone+classifier
      - Correct PA batch size (8 per source + 24 target = 48 total) which
        keeps per-step gradient magnitudes 8x smaller than the A100 batch sizes
        that caused the original explosion.
    """
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.grl = GradientReversalLayer()
        # Exact spec architecture: Linear→ReLU→Dropout(0.5)→Linear
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Full forward with GRL: use during backbone+classifier update (Pass 2)."""
        x = self.grl(x, alpha)
        return self.net(x)
