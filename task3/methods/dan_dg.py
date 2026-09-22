"""task3/methods/dan_dg.py — DAN-DG (multi-source pairwise MMD alignment) for Task 3."""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from common.metrics import calculate_metrics


def compute_mmd(source_features: torch.Tensor, target_features: torch.Tensor, kernel_scales: list[float]) -> torch.Tensor:
    """Exact vectorized MMD computation matching Task 2."""
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


class DanDGTrainer:
    def __init__(self, backbone, classifier, config, device):
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.criterion = nn.CrossEntropyLoss()

    def _freeze_bn(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, source_loaders, val_loaders):
        # Allow loaders passed as dict or list
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
        history = {"train_loss": [], "align_loss": [], "val_macro_f1": []}
        checkpoint_path = self.config.get(
            "checkpoint_path",
            os.path.join(self.config.get("output_dir", "task3/results"), "dan_dg_checkpoint.pth"),
        )
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        lambda_dg = float(self.config.get("lambda_dg", 1.0))
        kernel_scales = self.config.get("kernel_scales", [0.5, 1.0, 2.0])

        epoch_bar = tqdm(range(self.config["max_epochs"]), desc="dan_dg", unit="epoch")
        for epoch in epoch_bar:
            self.backbone.train()
            self.classifier.train()
            self._freeze_bn()

            total_cls = 0.0
            total_align = 0.0
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

                # Concatenate all domains for unified forward pass
                counts = [bx.size(0) for bx in batch_x]
                x_all = torch.cat(batch_x, dim=0).to(self.device)
                y_all = torch.cat(batch_y, dim=0).to(self.device)

                self.optimizer.zero_grad()
                features = self.backbone(x_all)
                logits = self.classifier(features)

                cls_loss = self.criterion(logits, y_all)

                # Split features back by domain
                feats_split = torch.split(features, counts, dim=0)
                # Pairwise MMD across observed sources: (0, 1), (0, 2), (1, 2)
                mmd_pairs = []
                for i in range(len(feats_split)):
                    for j in range(i + 1, len(feats_split)):
                        mmd_pairs.append(compute_mmd(feats_split[i], feats_split[j], kernel_scales))

                avg_mmd = sum(mmd_pairs) / len(mmd_pairs) if mmd_pairs else torch.tensor(0.0, device=self.device)
                loss = cls_loss + lambda_dg * avg_mmd

                loss.backward()
                self.optimizer.step()

                total_cls += cls_loss.item()
                total_align += avg_mmd.item()
                batch_bar.set_postfix(cls=f"{cls_loss.item():.4f}", mmd=f"{avg_mmd.item():.4f}")

            avg_cls = total_cls / n_iters
            avg_align = total_align / n_iters
            history["train_loss"].append(avg_cls)
            history["align_loss"].append(avg_align)

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
            epoch_bar.set_postfix(cls=f"{avg_cls:.4f}", mmd=f"{avg_align:.4f}", val_f1=f"{mean_val_f1:.4f}")

            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, mean_val_f1)
                tqdm.write(f"  [dan_dg] Epoch {epoch+1}: cls={avg_cls:.4f}  mmd={avg_align:.4f}  val_f1={mean_val_f1:.4f}  ✓ new best")
            else:
                patience_counter += 1
                tqdm.write(f"  [dan_dg] Epoch {epoch+1}: cls={avg_cls:.4f}  mmd={avg_align:.4f}  val_f1={mean_val_f1:.4f}  (patience {patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config["patience"]:
                    tqdm.write(f"  [dan_dg] Early stopping at epoch {epoch+1}.")
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

