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
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(self.config['n_epochs']):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['n_epochs']}")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                bs = x.size(0)
                half = bs // 2
                x1, y1 = x[:half], y[:half]
                x2, y2 = x[half:], y[half:]
                
                self.optimizer.zero_grad()
                
                # First half: L_cp
                logits_15 = self.model(x1)
                l_cp = self._classifier_placeholder_loss(logits_15, y1)
                
                # Second half: L_dp
                if x2.size(0) > 1:
                    idx = torch.randperm(x2.size(0), device=self.device)
                    # Simple heuristic for differing labels
                    shift = torch.randint(1, 10, (x2.size(0),), device=self.device)
                    x_j = x2[idx]
                    logits_mixed, _ = manifold_mixup_forward(self.model, x2, x_j, alpha=2.0)
                    l_dp = self._data_placeholder_loss(logits_mixed)
                else:
                    l_dp = 0.0
                    
                loss = l_cp + l_dp
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * bs
                pbar.set_postfix({'loss': loss.item()})
                
            self.scheduler.step()
            train_loss /= len(train_loader.dataset)
            
            val_acc = self.evaluate(val_loader)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(self.config['checkpoint_path']), exist_ok=True)
                torch.save(self.model.state_dict(), self.config['checkpoint_path'])
                
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})")
            
        return history
        
    def _classifier_placeholder_loss(self, logits_15, y):
        z_known = logits_15[:, :10]
        z_dummy = logits_15[:, 10:]
        ce = self.criterion(z_known, y)
        
        max_dummy, _ = z_dummy.max(dim=1)
        z_known_true = z_known[torch.arange(y.size(0)), y]
        penalty = F.relu(max_dummy - z_known_true).mean()
        
        return ce + self.config['beta'] * penalty
        
    def _data_placeholder_loss(self, logits_mixed):
        bs = logits_mixed.size(0)
        random_dummy_label = torch.randint(0, 5, (bs,), device=self.device) + 10
        return self.config['gamma'] * self.criterion(logits_mixed, random_dummy_label)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds = logits[:, :10].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
