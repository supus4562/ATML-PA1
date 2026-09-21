import torch
import torch.nn as nn
from torch.optim import AdamW
import json
import os
import math
from common.metrics import calculate_accuracy, calculate_macro_f1

class SAMTrainer:
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

    def _sam_step(self, X, Y):
        self.optimizer.zero_grad()
        logits = self.classifier(self.backbone(X))
        loss = self.criterion(logits, Y)
        loss.backward()
        
        global_norm = 0.0
        for param in self.backbone.parameters():
            if param.grad is not None:
                global_norm += param.grad.norm(2).item() ** 2
        for param in self.classifier.parameters():
            if param.grad is not None:
                global_norm += param.grad.norm(2).item() ** 2
        global_norm = math.sqrt(global_norm)
        
        eps = {}
        rho = self.config['rho']
        scale = rho / (global_norm + 1e-12)
        
        for name, param in self.backbone.named_parameters():
            if param.grad is not None:
                e = param.grad * scale
                eps[f"backbone_{name}"] = e
                param.data.add_(e)
        for name, param in self.classifier.named_parameters():
            if param.grad is not None:
                e = param.grad * scale
                eps[f"classifier_{name}"] = e
                param.data.add_(e)
                
        self.optimizer.zero_grad()
        logits = self.classifier(self.backbone(X))
        loss2 = self.criterion(logits, Y)
        loss2.backward()
        
        for name, param in self.backbone.named_parameters():
            if param.grad is not None:
                param.data.sub_(eps[f"backbone_{name}"])
        for name, param in self.classifier.named_parameters():
            if param.grad is not None:
                param.data.sub_(eps[f"classifier_{name}"])
                
        self.optimizer.step()
        
        return loss.item()

    def train(self, source_loaders, val_loaders):
        best_val_f1 = 0.0
        epochs_no_improve = 0
        history = []
        rho_str = f"rho{self.config['rho']}"
        
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
                
                loss = self._sam_step(X, Y)
                epoch_loss += loss
                
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
                self.save_checkpoint(os.path.join(self.config['output_dir'], f"sam_{rho_str}_checkpoint.pth"), epoch, mean_f1)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config['patience']:
                    break
                    
        with open(os.path.join(self.config['output_dir'], f"sam_{rho_str}_training_history.json"), 'w') as f:
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
