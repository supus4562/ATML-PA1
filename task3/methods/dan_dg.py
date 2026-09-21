import torch
import torch.nn as nn
from torch.optim import AdamW
import json
import os
from common.metrics import calculate_accuracy, calculate_macro_f1

class DanDGTrainer:
    def __init__(self, backbone, classifier, config, device):
        self.backbone = backbone.to(device)
        self.classifier = classifier.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=float(config['lr']),
            weight_decay=float(config['weight_decay'])
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def _freeze_bn(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _compute_mmd_across_domains(self, features_per_domain):
        def mmd2(X, Y):
            n_x, n_y = X.shape[0], Y.shape[0]
            XX = torch.cdist(X, X, p=2) ** 2
            YY = torch.cdist(Y, Y, p=2) ** 2
            XY = torch.cdist(X, Y, p=2) ** 2
            
            with torch.no_grad():
                combined = torch.cat([X, Y], dim=0)
                dists = torch.cdist(combined, combined, p=2) ** 2
                median_dist = torch.median(dists[dists > 0])
                if median_dist == 0:
                    median_dist = torch.tensor(1.0, device=self.device)
            
            bandwidths = [scale * median_dist for scale in self.config['kernel_scales']]
            
            def rbf(D, bandwidths):
                return sum(torch.exp(-D / bw) for bw in bandwidths)
            
            K_XX = rbf(XX, bandwidths)
            K_YY = rbf(YY, bandwidths)
            K_XY = rbf(XY, bandwidths)
            
            return K_XX.sum() / (n_x * n_x) + K_YY.sum() / (n_y * n_y) - 2 * K_XY.sum() / (n_x * n_y)

        f_photo, f_art, f_cartoon = features_per_domain
        mmd_pa = mmd2(f_photo, f_art)
        mmd_pc = mmd2(f_photo, f_cartoon)
        mmd_ac = mmd2(f_art, f_cartoon)
        
        return (mmd_pa + mmd_pc + mmd_ac) / 3.0

    def train(self, source_loaders, val_loaders):
        best_val_f1 = 0.0
        epochs_no_improve = 0
        history = []
        
        for epoch in range(self.config['max_epochs']):
            self.backbone.train()
            self.classifier.train()
            self._freeze_bn()
            
            iterators = [iter(loader) for loader in source_loaders.values()]
            epoch_loss = 0.0
            
            min_batches = min(len(loader) for loader in source_loaders.values())
            
            for _ in range(min_batches):
                batches = [next(it) for it in iterators]
                X = torch.cat([b[0] for b in batches]).to(self.device)
                Y = torch.cat([b[1] for b in batches]).to(self.device)
                
                self.optimizer.zero_grad()
                features = self.backbone(X)
                logits = self.classifier(features)
                
                loss_cls = self.criterion(logits, Y)
                
                batch_size = X.shape[0] // 3
                f_photo = features[:batch_size]
                f_art = features[batch_size:2*batch_size]
                f_cartoon = features[2*batch_size:]
                
                loss_align = self.config['lambda_dg'] * self._compute_mmd_across_domains([f_photo, f_art, f_cartoon])
                
                loss = loss_cls + loss_align
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            val_results = self._evaluate(val_loaders)
            mean_f1 = sum(v['macro_f1'] for v in val_results.values()) / len(val_results)
            
            history.append({
                'epoch': epoch,
                'train_loss': epoch_loss / min_batches,
                'val_metrics': val_results,
                'mean_f1': mean_f1
            })
            
            if mean_f1 > best_val_f1:
                best_val_f1 = mean_f1
                epochs_no_improve = 0
                self.save_checkpoint(os.path.join(self.config['output_dir'], f"{self.config['method']}_checkpoint.pth"), epoch, mean_f1)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config['patience']:
                    break
                    
        with open(os.path.join(self.config['output_dir'], f"{self.config['method']}_training_history.json"), 'w') as f:
            json.dump(history, f, indent=4)
            
        return history

    def _evaluate(self, val_loaders):
        self.backbone.eval()
        self.classifier.eval()
        results = {}
        with torch.no_grad():
            for name, loader in val_loaders.items():
                all_preds, all_labels = [], []
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.classifier(self.backbone(x))
                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(y.cpu())
                
                preds = torch.cat(all_preds).numpy()
                labels = torch.cat(all_labels).numpy()
                results[name] = {
                    'accuracy': calculate_accuracy(preds, labels),
                    'macro_f1': calculate_macro_f1(preds, labels)
                }
        return results

    def save_checkpoint(self, path, epoch, val_metric):
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'head_state_dict': self.classifier.state_dict(),
            'epoch': epoch,
            'val_metric': val_metric,
            'config': self.config
        }, path)
