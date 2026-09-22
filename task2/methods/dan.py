"""task2/methods/dan.py — DAN (MMD alignment) trainer for Task 2."""
from __future__ import annotations
import torch
import torch.nn as nn
from tqdm import tqdm
from common.metrics import calculate_metrics


def compute_mmd(source_features, target_features, kernel_scales):
    n = source_features.size(0)
    m = target_features.size(0)
    combined = torch.cat([source_features, target_features], dim=0)
    xx = torch.sum(combined ** 2, dim=1, keepdim=True)
    dist = xx + xx.t() - 2.0 * torch.matmul(combined, combined.t())
    median_dist = torch.median(dist[dist > 0])
    if median_dist == 0:
        median_dist = torch.tensor(1.0, device=combined.device)
    mmd2 = torch.tensor(0.0, device=combined.device)
    for scale in kernel_scales:
        bandwidth = scale * median_dist
        kernel_val = torch.exp(-dist / (2.0 * bandwidth))
        k_ss = kernel_val[:n, :n]
        k_tt = kernel_val[n:, n:]
        k_st = kernel_val[:n, n:]
        mmd2 = mmd2 + (
            (torch.sum(k_ss) - torch.trace(k_ss)) / (n * (n - 1))
            + (torch.sum(k_tt) - torch.trace(k_tt)) / (m * (m - 1))
            - 2.0 * torch.mean(k_st)
        )
    return mmd2


class DANTrainer:
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

    def train(self, source_loaders, target_loader, val_loaders):
        best_val_f1 = 0.0
        patience_counter = 0
        history = {"train_loss": [], "align_loss": [], "val_macro_f1": []}
        checkpoint_path = self.config.get("checkpoint_path", "task2/results/dan_checkpoint.pth")
        lambda_mmd = self.config["lambda_mmd"]

        epoch_bar = tqdm(range(self.config["max_epochs"]), desc="dan", unit="epoch")
        for epoch in epoch_bar:
            self.backbone.train()
            self.classifier.train()
            self.backbone.freeze_bn()
            total_loss = 0.0
            total_align = 0.0
            iters = max(len(dl) for dl in source_loaders)
            source_iters = [iter(dl) for dl in source_loaders]
            target_iter = iter(target_loader)

            batch_bar = tqdm(range(iters), desc="  batches", leave=False, unit="batch")
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
                try:
                    tx, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tx, _ = next(target_iter)
                sx = torch.cat(batch_x, dim=0).to(self.device)
                sy = torch.cat(batch_y, dim=0).to(self.device)
                tx = tx.to(self.device)
                self.optimizer.zero_grad()
                s_feat = self.backbone(sx)
                t_feat = self.backbone(tx)
                logits = self.classifier(s_feat)
                cls_loss = self.criterion(logits, sy)
                mmd_loss = compute_mmd(s_feat, t_feat, self.config["kernel_scales"])
                loss = cls_loss + lambda_mmd * mmd_loss
                loss.backward()
                self.optimizer.step()
                total_loss  += cls_loss.item()
                total_align += mmd_loss.item()
                batch_bar.set_postfix(cls=f"{cls_loss.item():.4f}", mmd=f"{mmd_loss.item():.4f}")

            avg_loss  = total_loss  / iters
            avg_align = total_align / iters
            history["train_loss"].append(avg_loss)
            history["align_loss"].append(avg_align)

            self.backbone.eval()
            self.classifier.eval()
            val_f1s = []
            with torch.no_grad():
                for dl in val_loaders:
                    preds, targets = [], []
                    for x, y in dl:
                        x, y = x.to(self.device), y.to(self.device)
                        preds.append(self.classifier(self.backbone(x)).argmax(dim=1))
                        targets.append(y)
                    m = calculate_metrics(torch.cat(targets).cpu(), torch.cat(preds).cpu())
                    val_f1s.append(m["macro_f1"])

            mean_val_f1 = sum(val_f1s) / len(val_f1s)
            history["val_macro_f1"].append(mean_val_f1)
            epoch_bar.set_postfix(cls=f"{avg_loss:.4f}", mmd=f"{avg_align:.4f}", val_f1=f"{mean_val_f1:.4f}")

            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, mean_val_f1)
                tqdm.write(f"  [dan] Epoch {epoch+1}: cls={avg_loss:.4f}  mmd={avg_align:.4f}  val_f1={mean_val_f1:.4f}  ✓ new best")
            else:
                patience_counter += 1
                tqdm.write(f"  [dan] Epoch {epoch+1}: cls={avg_loss:.4f}  mmd={avg_align:.4f}  val_f1={mean_val_f1:.4f}  (patience {patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config["patience"]:
                    tqdm.write(f"  [dan] Early stopping at epoch {epoch+1}.")
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
