import torch
import torch.nn as nn
from tqdm import tqdm
from common.metrics import calculate_metrics

def compute_mmd(source_features, target_features, kernel_scales):
    n = source_features.size(0)
    m = target_features.size(0)
    combined = torch.cat([source_features, target_features], dim=0)
    
    # pairwise squared distances
    xx = torch.sum(combined**2, dim=1, keepdim=True)
    yy = xx.t()
    dist = xx + yy - 2.0 * torch.matmul(combined, combined.t())
    
    median_dist = torch.median(dist[dist > 0])
    if median_dist == 0:
        median_dist = 1.0
        
    mmd2 = 0
    for scale in kernel_scales:
        bandwidth = scale * median_dist
        kernel_val = torch.exp(-dist / (2.0 * bandwidth))
        
        k_ss = kernel_val[:n, :n]
        k_tt = kernel_val[n:, n:]
        k_st = kernel_val[:n, n:]
        
        # Unbiased estimators
        mmd2 += (torch.sum(k_ss) - torch.trace(k_ss)) / (n * (n - 1)) \
              + (torch.sum(k_tt) - torch.trace(k_tt)) / (m * (m - 1)) \
              - 2.0 * torch.mean(k_st)
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
            lr=config['lr'], weight_decay=config['weight_decay']
        )

    def train(self, source_loaders, target_loader, val_loaders):
        best_val_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'align_loss': [], 'val_macro_f1': []}
        
        for epoch in range(self.config['max_epochs']):
            self.backbone.train()
            self.classifier.train()
            self.backbone.freeze_bn()
            
            total_loss = 0
            total_align = 0
            iters = max([len(dl) for dl in source_loaders])
            source_iters = [iter(dl) for dl in source_loaders]
            target_iter = iter(target_loader)
            
            for _ in tqdm(range(iters), desc=f'Epoch {epoch+1}'):
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
                
                logits = self.classifier(s_feat)
                cls_loss = self.criterion(logits, sy)
                
                mmd_loss = compute_mmd(s_feat, t_feat, self.config['kernel_scales'])
                loss = cls_loss + self.config['lambda_mmd'] * mmd_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += cls_loss.item()
                total_align += mmd_loss.item()
                
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
            'epoch': epoch,
            'val_macro_f1': val_metric,
            'config': self.config
        }, path)
