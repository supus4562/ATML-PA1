"""task3/methods/sam.py — Sharpness-Aware Minimization (SAM) trainer for Task 3."""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from common.metrics import calculate_metrics


class SAMTrainer:
    def __init__(self, backbone, classifier, config, device):
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.config = config
        self.device = device
        self.params = [
            p for p in list(self.backbone.parameters()) + list(self.classifier.parameters())
            if p.requires_grad
        ]
        self.optimizer = AdamW(
            self.params,
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.rho = float(config.get("rho", 0.05))

    def _freeze_bn(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _sam_step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        # First pass: compute loss and gradients at theta
        self.optimizer.zero_grad()
        self._freeze_bn()
        logits = self.classifier(self.backbone(X))
        loss = self.criterion(logits, Y)
        loss.backward()

        # Compute global gradient norm entirely on GPU (no CPU sync)
        grads = [p.grad for p in self.params if p.grad is not None]
        if grads:
            grad_norm = torch.norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            scale = self.rho / (grad_norm + 1e-12)
        else:
            scale = 0.0

        # Perturb parameters: theta = theta + eps
        eps_list = []
        for p in self.params:
            if p.grad is not None:
                e = p.grad * scale
                eps_list.append(e)
                p.data.add_(e)
            else:
                eps_list.append(None)

        # Second pass: compute gradients at perturbed point (BN frozen)
        self.optimizer.zero_grad()
        self._freeze_bn()
        logits2 = self.classifier(self.backbone(X))
        loss2 = self.criterion(logits2, Y)
        loss2.backward()

        # Restore parameters: theta = theta - eps
        for p, e in zip(self.params, eps_list):
            if e is not None:
                p.data.sub_(e)

        self.optimizer.step()
        return loss.item()

    def train(self, source_loaders, val_loaders):
        if isinstance(source_loaders, dict):
            domains = self.config.get("source_domains", list(source_loaders.keys()))
            src_loaders = [source_loaders[d] for d in domains]
        else:
            src_loaders = list(source_loaders)

        if isinstance(val_loaders, dict):
            val_ldrs = list(val_loaders.values())
        else:
            val_ldrs = list(val_loaders)

        best_val_f1 = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_macro_f1": []}
        checkpoint_path = self.config.get(
            "checkpoint_path",
            os.path.join(self.config.get("output_dir", "task3/results"), f"sam_rho{self.rho}_checkpoint.pth"),
        )
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

        epoch_bar = tqdm(range(self.config["max_epochs"]), desc=f"sam (ρ={self.rho})", unit="epoch")
        for epoch in epoch_bar:
            self.backbone.train()
            self.classifier.train()
            self._freeze_bn()

            total_loss = 0.0
            n_iters = max(len(loader) for loader in src_loaders)
            source_iters = [iter(loader) for loader in src_loaders]

            batch_bar = tqdm(range(n_iters), desc="  batches", leave=False, unit="batch")
            for _ in batch_bar:
                batch_x, batch_y = [], []
                for i, (siter, dl) in enumerate(zip(source_iters, src_loaders)):
                    try:
                        x, y = next(siter)
                    except StopIteration:
                        source_iters[i] = iter(dl)
                        x, y = next(source_iters[i])
                    batch_x.append(x)
                    batch_y.append(y)

                x_all = torch.cat(batch_x, dim=0).to(self.device)
                y_all = torch.cat(batch_y, dim=0).to(self.device)

                loss = self._sam_step(x_all, y_all)
                total_loss += loss
                batch_bar.set_postfix(loss=f"{loss:.4f}")

            avg_loss = total_loss / n_iters
            history["train_loss"].append(avg_loss)

            # ── Validation ────────────────────────────────────────────────────
            self.backbone.eval()
            self.classifier.eval()
            val_f1s = []
            with torch.no_grad():
                for dl in val_ldrs:
                    preds, targets = [], []
                    for x, y in dl:
                        x, y = x.to(self.device), y.to(self.device)
                        preds.append(self.classifier(self.backbone(x)).argmax(dim=1).cpu())
                        targets.append(y.cpu())
                    m = calculate_metrics(torch.cat(targets), torch.cat(preds))
                    val_f1s.append(m["macro_f1"])

            mean_val_f1 = sum(val_f1s) / len(val_f1s)
            history["val_macro_f1"].append(mean_val_f1)
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val_f1=f"{mean_val_f1:.4f}")

            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, mean_val_f1)
                tqdm.write(f"  [sam ρ={self.rho}] Epoch {epoch+1}: loss={avg_loss:.4f}  val_f1={mean_val_f1:.4f}  ✓ new best")
            else:
                patience_counter += 1
                tqdm.write(f"  [sam ρ={self.rho}] Epoch {epoch+1}: loss={avg_loss:.4f}  val_f1={mean_val_f1:.4f}  (patience {patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config["patience"]:
                    tqdm.write(f"  [sam ρ={self.rho}] Early stopping at epoch {epoch+1}.")
                    break

        return history

    def save_checkpoint(self, path: str, epoch: int, val_metric: float) -> None:
        torch.save({
            "backbone_state_dict": self.backbone.state_dict(),
            "head_state_dict":     self.classifier.state_dict(),
            "epoch":               epoch,
            "val_macro_f1":        val_metric,
            "config":              self.config,
        }, path)

