import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from task4.models.resnet_cifar import ResNetCIFAR
from task4.methods.manifold_mixup import manifold_mixup_forward
from tqdm import tqdm
import os

class PROSERTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = ResNetCIFAR(num_classes=10).to(device)
        self.model.load_state_dict(torch.load(config['vanilla_checkpoint']))
        
        # Replace FC layer to output 15 (10 known + 5 dummy)
        old_fc = self.model.fc
        self.model.fc = nn.Linear(512, 15).to(device)
        self.model.fc.weight.data[:10] = old_fc.weight.data
        self.model.fc.bias.data[:10] = old_fc.bias.data
        nn.init.uniform_(self.model.fc.weight.data[10:], -0.01, 0.01)
        nn.init.uniform_(self.model.fc.bias.data[10:], -0.01, 0.01)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'], 
                                   momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['n_epochs'])
        
    def train(self, train_loader, val_loader):
        best_acc = 0.0
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        n_epochs = self.config['n_epochs']
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{n_epochs}")
            for x, y in pbar:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                bs = x.size(0)
                half = bs // 2
                x1, y1 = x[:half], y[:half]
                x2, y2 = x[half:], y[half:]
                
                self.optimizer.zero_grad()
                
                # First half: L_cp (Classifier Placeholders)
                logits_15 = self.model(x1)
                l_cp = self._classifier_placeholder_loss(logits_15, y1)
                
                # Second half: L_dp (Data Placeholders via Manifold Mixup with y_i != y_j)
                l_dp = torch.tensor(0.0, device=self.device)
                if x2.size(0) > 1:
                    # Form pairs such that y_i != y_j
                    n2 = x2.size(0)
                    perm = torch.randperm(n2, device=self.device)
                    # Shift perm until y2 != y2[perm] where possible
                    for _ in range(n2):
                        same = (y2 == y2[perm])
                        if not same.any():
                            break
                        perm[same] = torch.roll(perm, shifts=1)[same]
                    
                    diff_mask = (y2 != y2[perm])
                    if diff_mask.any():
                        x_i = x2[diff_mask]
                        x_j = x2[perm][diff_mask]
                        logits_mixed, _ = manifold_mixup_forward(self.model, x_i, x_j, alpha=2.0)
                        l_dp = self._data_placeholder_loss(logits_mixed)
                    
                loss = l_cp + l_dp
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * bs
                preds = logits_15[:, :10].argmax(dim=1)
                train_correct += (preds == y1).sum().item()
                train_total += y1.size(0)
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.5f}"})
                
            self.scheduler.step()
            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / max(1, train_total)
            
            val_loss, val_acc = self.evaluate(val_loader)
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(self.config['checkpoint_path']), exist_ok=True)
                torch.save(self.model.state_dict(), self.config['checkpoint_path'])
                
            tqdm.write(
                f"[Epoch {epoch+1:03d}/{n_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} (Best: {best_acc:.4f}) | LR: {current_lr:.6f}"
            )
            
        return history
        
    def _classifier_placeholder_loss(self, logits_15, y):
        # Known class CE loss
        z_known = logits_15[:, :10]
        z_dummy = logits_15[:, 10:]
        ce = self.criterion(z_known, y)
        
        # When correct class is excluded, dummy should be stronger than remaining known classes:
        # z_dummy > max_{k != y} z_k
        max_dummy, _ = z_dummy.max(dim=1)
        z_known_masked = z_known.clone()
        z_known_masked[torch.arange(y.size(0), device=self.device), y] = -1e9
        max_other_known, _ = z_known_masked.max(dim=1)
        penalty = F.relu(max_other_known - max_dummy + 1.0).mean()
        
        return ce + self.config.get('beta', 1.0) * penalty
        
    def _data_placeholder_loss(self, logits_mixed):
        bs = logits_mixed.size(0)
        # Train synthetic proxy unknowns toward the dummy classifiers
        random_dummy_label = torch.randint(10, 15, (bs,), device=self.device)
        return self.config.get('gamma', 0.1) * self.criterion(logits_mixed, random_dummy_label)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = self.model(x)
                # PDF: compute CSA using only the ten known-class logits
                loss = self.criterion(logits[:, :10], y)
                total_loss += loss.item() * x.size(0)
                preds = logits[:, :10].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return total_loss / total, correct / total
