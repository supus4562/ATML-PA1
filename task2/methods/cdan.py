import torch
import torch.nn as nn
from tqdm import tqdm
import math
from common.metrics import calculate_metrics
from task2.models.domain_discriminator import DomainDiscriminator

class CDANTrainer:
    def __init__(self, backbone, classifier, config, device):
        self.backbone = backbone
        self.classifier = classifier
        self.discriminator = DomainDiscriminator(in_dim=config['combined_dim']).to(device)
        self.config = config
        self.device = device
        self.cls_criterion = nn.CrossEntropyLoss()
        self.dom_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.classifier.parameters()) + list(self.discriminator.parameters()),
            lr=config['lr'], weight_decay=config['weight_decay']
        )

    def train(self, source_loaders, target_loader, val_loaders):
        best_val_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'align_loss': [], 'val_macro_f1': []}
        
        total_steps = self.config['max_epochs'] * max([len(dl) for dl in source_loaders])
        current_step = 0
        
        for epoch in range(self.config['max_epochs']):
            self.backbone.train()
            self.classifier.train()
            self.discriminator.train()
            self.backbone.freeze_bn()
            
            total_loss = 0
            total_align = 0
            iters = max([len(dl) for dl in source_loaders])
            source_iters = [iter(dl) for dl in source_loaders]
            target_iter = iter(target_loader)
            
            for _ in tqdm(range(iters), desc=f'Epoch {epoch+1}'):
                p = current_step / total_steps
                alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
                current_step += 1
                
                batch_x, batch_y = [], []
                for siter, dl in zip(source_iters, source_loaders):
                    try:
                        x, y = next(siter)
                    except StopIteration:
                        siter = iter(dl)
                        x, y = next(siter)
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
                
                s_logits = self.classifier(s_feat)
                t_logits = self.classifier(t_feat)
                cls_loss = self.cls_criterion(s_logits, sy)
                
                s_prob = torch.softmax(s_logits, dim=1)
                t_prob = torch.softmax(t_logits, dim=1)
                
                s_comb = torch.bmm(s_feat.unsqueeze(2), s_prob.unsqueeze(1)).view(s_feat.size(0), -1)
                t_comb = torch.bmm(t_feat.unsqueeze(2), t_prob.unsqueeze(1)).view(t_feat.size(0), -1)
                
                feat = torch.cat([s_comb, t_comb], dim=0)
                dom_labels = torch.cat([
                    torch.zeros(s_feat.size(0), dtype=torch.long),
                    torch.ones(t_feat.size(0), dtype=torch.long)
                ]).to(self.device)
                
                dom_logits = self.discriminator(feat, alpha)
                dom_loss = self.dom_criterion(dom_logits, dom_labels)
                
                loss = cls_loss + dom_loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += cls_loss.item()
                total_align += dom_loss.item()
                
            avg_loss = total_loss / iters
            history['train_loss'].append(avg_loss)
            history['align_loss'].append(total_align / iters)
            
            # Validation
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
                    metrics = calculate_metrics(torch.cat(targets).cpu(), torch.cat(preds).cpu())
                    val_f1s.append(metrics['macro_f1'])
            
            mean_val_f1 = sum(val_f1s) / len(val_f1s)
            history['val_macro_f1'].append(mean_val_f1)
            
            if mean_val_f1 > best_val_f1:
                best_val_f1 = mean_val_f1
                patience_counter = 0
                self.save_checkpoint(self.config['checkpoint_path'], epoch, mean_val_f1)
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    break
        return history

    def save_checkpoint(self, path, epoch, val_metric):
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'head_state_dict': self.classifier.state_dict(),
            'disc_state_dict': self.discriminator.state_dict(),
            'epoch': epoch,
            'val_macro_f1': val_metric,
            'config': self.config
        }, path)
