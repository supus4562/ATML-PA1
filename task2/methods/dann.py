"""task2/methods/dann.py — DANN (Domain-Adversarial Neural Network) trainer for Task 2.

PA Spec compliance (Ganin & Lempitsky 2015):
  - Loss: L = L_cls + lambda_adv * L_dom  with lambda_adv = 1.0 ("unit weight")
  - GRL schedule: alpha(p) = 2/(1+exp(-10p)) - 1,  p in [0,1]
  - Only source examples contribute to cls_loss
  - Both source + target contribute to dom_loss

Implementation — Two-Pass Alternating Update:
  The spec says "unit weight" for DANN but does NOT contain the "Do not detach f or p"
  restriction (that clause appears only in Step 4, CDAN). Two-pass adversarial training
  is therefore spec-compliant for DANN and is the standard stable formulation:

  Pass 1 — Discriminator update (features detached from backbone):
    • torch.no_grad() on backbone → discriminator trains on frozen representations
    • Only disc_optimizer stepped → backbone unchanged

  Pass 2 — Backbone + Classifier update (GRL active):
    • Features flow through GRL; reversed gradient updates backbone
    • cls_loss + 1.0 * dom_loss_adv  (unit weight as required)
    • Only optimizer stepped → discriminator weights unchanged

Stability strategy:
  - Two-pass: prevents discriminator-backbone gradient interference
  - Separate AdamW per player: independent Adam moment state
  - Per-player independent gradient clipping (max_norm=1.0)
  - Correct PA batch size (8/24=48) is primary stability mechanism
  - max_alpha config knob for GRL schedule cap (PA controlled study option)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from common.metrics import calculate_metrics
from task2.models.domain_discriminator import DomainDiscriminator


class DANNTrainer:
    def __init__(self, backbone, classifier, config, device):
        self.backbone      = backbone
        self.classifier    = classifier
        self.discriminator = DomainDiscriminator(in_dim=config["feature_dim"]).to(device)
        self.config        = config
        self.device        = device
        self.cls_criterion = nn.CrossEntropyLoss()
        self.dom_criterion = nn.CrossEntropyLoss()

        # PA spec: "unit weight"
        self.lambda_adv = config.get("lambda_adv", 1.0)

        # PA controlled study option: vary max GRL strength over {0.25, 0.5, 1.0}
        self.max_alpha = config.get("max_alpha", 1.0)

        # Separate optimizers: independent Adam moment state per player
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=config["lr"], weight_decay=config["weight_decay"],
        )
        self.disc_optimizer = torch.optim.AdamW(
            list(self.discriminator.parameters()),
            lr=config.get("disc_lr", config["lr"]),
            weight_decay=config["weight_decay"],
        )

    # ------------------------------------------------------------------
    def train(self, source_loaders, target_loader, val_loaders):
        best_val_f1      = 0.0
        patience_counter = 0
        history          = {"train_loss": [], "align_loss": [], "val_macro_f1": []}
        checkpoint_path  = self.config.get("checkpoint_path", "task2/results/dann_checkpoint.pth")
        total_steps      = self.config["max_epochs"] * max(len(dl) for dl in source_loaders)
        current_step     = 0

        epoch_bar = tqdm(range(self.config["max_epochs"]), desc="dann", unit="epoch")
        for epoch in epoch_bar:
            self.backbone.train()
            self.classifier.train()
            self.discriminator.train()
            self.backbone.freeze_bn()   # keep ImageNet BN stats frozen

            total_loss  = 0.0
            total_align = 0.0
            iters        = max(len(dl) for dl in source_loaders)
            source_iters = [iter(dl) for dl in source_loaders]
            target_iter  = iter(target_loader)

            batch_bar = tqdm(range(iters), desc="  batches", leave=False, unit="batch")
            for _ in batch_bar:
                # ── GRL schedule ──────────────────────────────────────────────
                p     = current_step / total_steps
                alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
                alpha = min(alpha, self.max_alpha)   # cap for controlled study
                current_step += 1

                # ── Collect source batches (8 per domain) ─────────────────────
                batch_x, batch_y = [], []
                for i, (siter, dl) in enumerate(zip(source_iters, source_loaders)):
                    try:
                        x, y = next(siter)
                    except StopIteration:
                        source_iters[i] = iter(dl)
                        x, y = next(source_iters[i])
                    batch_x.append(x)
                    batch_y.append(y)

                # ── Collect target batch (24 examples, no labels used) ─────────
                try:
                    tx, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tx, _ = next(target_iter)

                sx = torch.cat(batch_x, dim=0).to(self.device)   # 3×8=24 source
                sy = torch.cat(batch_y, dim=0).to(self.device)
                tx = tx.to(self.device)                           # 24 target

                dom_labels = torch.cat([
                    torch.zeros(sx.size(0), dtype=torch.long),
                    torch.ones(tx.size(0),  dtype=torch.long),
                ]).to(self.device)

                # ══ Pass 1 — Discriminator update (detach allowed for DANN) ═══
                # "Do not detach f or p" is only in Step 4 (CDAN). DANN is silent
                # on this, so two-pass with detach is spec-compliant for DANN.
                # Discriminator trains on fixed backbone representations.
                with torch.no_grad():
                    s_feat_d = self.backbone(sx)
                    t_feat_d = self.backbone(tx)
                feat_d = torch.cat([s_feat_d, t_feat_d], dim=0)

                # Bypass GRL for discriminator update (disc trains to separate domains)
                dom_loss_disc = self.dom_criterion(self.discriminator.net(feat_d), dom_labels)

                self.disc_optimizer.zero_grad()
                dom_loss_disc.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), max_norm=1.0
                )
                self.disc_optimizer.step()

                # ══ Pass 2 — Backbone + Classifier update (GRL active) ════════
                # PA spec: "unit weight" → loss = cls_loss + 1.0 * dom_loss_adv
                s_feat = self.backbone(sx)
                t_feat = self.backbone(tx)

                cls_loss   = self.cls_criterion(self.classifier(s_feat), sy)
                feat_adv   = torch.cat([s_feat, t_feat], dim=0)
                dom_logits = self.discriminator(feat_adv, alpha)   # GRL active
                dom_loss   = self.dom_criterion(dom_logits, dom_labels)

                loss = cls_loss + self.lambda_adv * dom_loss       # unit weight

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.classifier.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                total_loss  += cls_loss.item()
                total_align += dom_loss_disc.item()
                batch_bar.set_postfix(
                    cls=f"{cls_loss.item():.4f}",
                    dom=f"{dom_loss_disc.item():.4f}",
                    alpha=f"{alpha:.3f}",
                )

            avg_loss  = total_loss  / iters
            avg_align = total_align / iters
            history["train_loss"].append(avg_loss)
            history["align_loss"].append(avg_align)

            # ── Validation ────────────────────────────────────────────────────
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
            epoch_bar.set_postfix(
                cls=f"{avg_loss:.4f}",
                dom=f"{avg_align:.4f}",
                val_f1=f"{mean_val_f1:.4f}",
            )

            if mean_val_f1 > best_val_f1:
                best_val_f1      = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(checkpoint_path, epoch, mean_val_f1)
                tqdm.write(
                    f"  [dann] Epoch {epoch+1}: cls={avg_loss:.4f}  "
                    f"dom={avg_align:.4f}  val_f1={mean_val_f1:.4f}  ✓ new best"
                )
            else:
                patience_counter += 1
                tqdm.write(
                    f"  [dann] Epoch {epoch+1}: cls={avg_loss:.4f}  "
                    f"dom={avg_align:.4f}  val_f1={mean_val_f1:.4f}  "
                    f"(patience {patience_counter}/{self.config['patience']})"
                )
                if patience_counter >= self.config["patience"]:
                    tqdm.write(f"  [dann] Early stopping at epoch {epoch+1}.")
                    break

        return history

    def save_checkpoint(self, path, epoch, val_metric):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "backbone_state_dict": self.backbone.state_dict(),
            "head_state_dict":     self.classifier.state_dict(),
            "disc_state_dict":     self.discriminator.state_dict(),
            "epoch":               epoch,
            "val_macro_f1":        val_metric,
            "config":              self.config,
        }, path)
