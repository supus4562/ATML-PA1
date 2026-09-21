"""task2/methods/source_only.py — Source-only ERM baseline for Task 2.

Trains on labeled source domain data only (no target data used).
Saves checkpoint to task2/results/source_only_checkpoint.pth.
This checkpoint is reused as the ERM baseline in Task 3.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from tqdm import tqdm

from common.metrics import calculate_metrics


class SourceOnlyTrainer:
    def __init__(self, backbone: nn.Module, classifier: nn.Module,
                 config: dict, device: torch.device) -> None:
        self.backbone = backbone
        self.classifier = classifier
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=config["lr"], weight_decay=config["weight_decay"],
        )

    def _freeze_bn(self) -> None:
        """Freeze all BatchNorm2d running stats (keep gamma/beta trainable)."""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train(self, source_loaders, target_loader, val_loaders) -> dict:
        """Train with domain-balanced source batches.

        Args:
            source_loaders: List of DataLoaders (one per source domain).
            target_loader:  Ignored (no target labels used).
            val_loaders:    List of DataLoaders for validation (one per source domain).
        Returns:
            Training history dict.
        """
        best_val_f1 = 0.0
        patience_counter = 0
        history: dict[str, list] = {"train_loss": [], "val_macro_f1": []}

        checkpoint_path = self.config.get(
            "checkpoint_path", "task2/results/source_only_checkpoint.pth"
        )
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

        for epoch in range(self.config["max_epochs"]):
            self.backbone.train()
            self.classifier.train()
            self._freeze_bn()

            total_loss = 0.0
            # Cycle iterators so shorter loaders wrap around
            source_iters = [iter(dl) for dl in source_loaders]
            n_iters = max(len(dl) for dl in source_loaders)

            for _ in tqdm(range(n_iters), desc=f"Epoch {epoch + 1}/{self.config['max_epochs']}",
                          leave=False):
                batch_x, batch_y = [], []
                for i, (siter, dl) in enumerate(zip(source_iters, source_loaders)):
                    try:
                        x, y = next(siter)
                    except StopIteration:
                        source_iters[i] = iter(dl)  # reset in-place
                        x, y = next(source_iters[i])
                    batch_x.append(x)
                    batch_y.append(y)

                x_all = torch.cat(batch_x, dim=0).to(self.device)
                y_all = torch.cat(batch_y, dim=0).to(self.device)

                self.optimizer.zero_grad()
                feats = self.backbone(x_all)
                logits = self.classifier(feats)
                loss = self.criterion(logits, y_all)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / n_iters
            history["train_loss"].append(avg_loss)

            # ── Validation ────────────────────────────────────────────────────
            self.backbone.eval()
            self.classifier.eval()
            val_f1s = []
            with torch.no_grad():
                for dl in val_loaders:
                    preds_list, targets_list = [], []
                    for x, y in dl:
                        x, y = x.to(self.device), y.to(self.device)
                        logits = self.classifier(self.backbone(x))
                        preds_list.append(logits.argmax(dim=1).cpu())
                        targets_list.append(y.cpu())
                    metrics = calculate_metrics(
                        torch.cat(targets_list), torch.cat(preds_list)
                    )
                    val_f1s.append(metrics["macro_f1"])

            mean_val_f1 = sum(val_f1s) / len(val_f1s)
            history["val_macro_f1"].append(mean_val_f1)
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}  val_macro_f1={mean_val_f1:.4f}")

            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, mean_val_f1)
                print(f"  ✓ New best ({best_val_f1:.4f}), checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    print(f"  Early stopping (patience={self.config['patience']}).")
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
