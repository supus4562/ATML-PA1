"""task2/methods/source_only.py — Source-only ERM baseline for Task 2."""
from __future__ import annotations
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from common.metrics import calculate_metrics


class SourceOnlyTrainer:
    def __init__(self, backbone, classifier, config, device):
        self.backbone = backbone
        self.classifier = classifier
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=config["lr"], weight_decay=config["weight_decay"],
        )

    def _freeze_bn(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def train(self, source_loaders, target_loader, val_loaders):
        best_val_f1 = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_macro_f1": []}
        checkpoint_path = self.config.get("checkpoint_path", "task2/results/source_only_checkpoint.pth")
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

        epoch_bar = tqdm(range(self.config["max_epochs"]), desc="source_only", unit="epoch")
        for epoch in epoch_bar:
            self.backbone.train()
            self.classifier.train()
            self._freeze_bn()

            total_loss = 0.0
            source_iters = [iter(dl) for dl in source_loaders]
            n_iters = max(len(dl) for dl in source_loaders)

            batch_bar = tqdm(range(n_iters), desc="  batches", leave=False, unit="batch")
            for _ in batch_bar:
                batch_x, batch_y = [], []
                for i, (siter, dl) in enumerate(zip(source_iters, source_loaders)):
                    try:
                        x, y = next(siter)
                    except StopIteration:
                        source_iters[i] = iter(dl)
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
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / n_iters
            history["train_loss"].append(avg_loss)

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
                    m = calculate_metrics(torch.cat(targets_list), torch.cat(preds_list))
                    val_f1s.append(m["macro_f1"])

            mean_val_f1 = sum(val_f1s) / len(val_f1s)
            history["val_macro_f1"].append(mean_val_f1)
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val_f1=f"{mean_val_f1:.4f}")

            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, mean_val_f1)
                tqdm.write(f"  [source_only] Epoch {epoch+1}: loss={avg_loss:.4f}  val_f1={mean_val_f1:.4f}  ✓ new best")
            else:
                patience_counter += 1
                tqdm.write(f"  [source_only] Epoch {epoch+1}: loss={avg_loss:.4f}  val_f1={mean_val_f1:.4f}  (patience {patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config["patience"]:
                    tqdm.write(f"  [source_only] Early stopping at epoch {epoch+1}.")
                    break
        return history

    def save_checkpoint(self, path, epoch, val_metric):
        torch.save({
            "backbone_state_dict": self.backbone.state_dict(),
            "head_state_dict":     self.classifier.state_dict(),
            "epoch":               epoch,
            "val_macro_f1":        val_metric,
            "config":              self.config,
        }, path)
