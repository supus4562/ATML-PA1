"""task2/methods/cdan.py — CDAN (class-conditional adversarial alignment) trainer for Task 2."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from common.metrics import calculate_metrics
from task2.models.domain_discriminator import DomainDiscriminator


class CDANTrainer:
    def __init__(self, backbone, classifier, config, device):
        self.backbone = backbone
        self.classifier = classifier
        self.discriminator = DomainDiscriminator(in_dim=config["combined_dim"]).to(device)
        self.config = config
        self.device = device
        self.cls_criterion = nn.CrossEntropyLoss()
        self.dom_criterion = nn.CrossEntropyLoss()
        self.lambda_adv = config.get("lambda_adv", 1.0)
        all_params = (
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + list(self.discriminator.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config["lr"], weight_decay=config["weight_decay"],
        )

    def train(self, source_loaders, target_loader, val_loaders):
        best_val_f1 = 0.0
        patience_counter = 0
        history = {"train_loss": [], "align_loss": [], "val_macro_f1": []}
        checkpoint_path = self.config.get("checkpoint_path", "task2/results/cdan_checkpoint.pth")
        total_steps = self.config["max_epochs"] * max(len(dl) for dl in source_loaders)
        current_step = 0

        epoch_bar = tqdm(range(self.config["max_epochs"]), desc="cdan", unit="epoch")
        for epoch in epoch_bar:
            self.backbone.train()
            self.classifier.train()
            self.discriminator.train()
            self.backbone.freeze_bn()
            total_loss = 0.0
            total_align = 0.0
            iters = max(len(dl) for dl in source_loaders)
            source_iters = [iter(dl) for dl in source_loaders]
            target_iter = iter(target_loader)

            batch_bar = tqdm(range(iters), desc="  batches", leave=False, unit="batch")
            for _ in batch_bar:
                p = current_step / total_steps
                alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
                current_step += 1
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
                dom_labels = torch.cat([
                    torch.zeros(sx.size(0), dtype=torch.long),
                    torch.ones(tx.size(0), dtype=torch.long),
                ]).to(self.device)

                # ── Single-pass forward with GRL (no detachment of f or p) ──
                s_feat  = self.backbone(sx)
                t_feat  = self.backbone(tx)
                s_logits = self.classifier(s_feat)
                t_logits = self.classifier(t_feat)
                cls_loss = self.cls_criterion(s_logits, sy)
                s_prob   = torch.softmax(s_logits, dim=1)
                t_prob   = torch.softmax(t_logits, dim=1)
                s_comb   = torch.bmm(s_feat.unsqueeze(2), s_prob.unsqueeze(1)).view(s_feat.size(0), -1)
                t_comb   = torch.bmm(t_feat.unsqueeze(2), t_prob.unsqueeze(1)).view(t_feat.size(0), -1)
                feat_adv = torch.cat([s_comb, t_comb], dim=0)
                dom_logits_adv = self.discriminator(feat_adv, alpha)  # GRL active
                dom_loss_adv   = self.dom_criterion(dom_logits_adv, dom_labels)
                loss = cls_loss + self.lambda_adv * dom_loss_adv
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters())
                    + list(self.classifier.parameters())
                    + list(self.discriminator.parameters()),
                    max_norm=5.0,
                )
                self.optimizer.step()

                total_loss  += cls_loss.item()
                total_align += dom_loss_adv.item()
                batch_bar.set_postfix(cls=f"{cls_loss.item():.4f}", dom=f"{dom_loss_adv.item():.4f}", alpha=f"{alpha:.3f}")

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
            epoch_bar.set_postfix(cls=f"{avg_loss:.4f}", dom=f"{avg_align:.4f}", val_f1=f"{mean_val_f1:.4f}")

            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, mean_val_f1)
                tqdm.write(f"  [cdan] Epoch {epoch+1}: cls={avg_loss:.4f}  dom={avg_align:.4f}  val_f1={mean_val_f1:.4f}  ✓ new best")
            else:
                patience_counter += 1
                tqdm.write(f"  [cdan] Epoch {epoch+1}: cls={avg_loss:.4f}  dom={avg_align:.4f}  val_f1={mean_val_f1:.4f}  (patience {patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config["patience"]:
                    tqdm.write(f"  [cdan] Early stopping at epoch {epoch+1}.")
                    break
        return history

    def save_checkpoint(self, path, epoch, val_metric):
        torch.save({
            "backbone_state_dict": self.backbone.state_dict(),
            "head_state_dict":     self.classifier.state_dict(),
            "disc_state_dict":     self.discriminator.state_dict(),
            "epoch":               epoch,
            "val_macro_f1":        val_metric,
            "config":              self.config,
        }, path)
