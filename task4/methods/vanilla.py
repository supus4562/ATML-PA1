import torch
import torch.nn as nn
import torch.optim as optim
from task4.models.resnet_cifar import ResNetCIFAR
import os
from tqdm import tqdm

class VanillaTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = ResNetCIFAR(num_classes=10).to(device)
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
                
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                
                bs = x.size(0)
                train_loss += loss.item() * bs
                preds = logits[:, :10].argmax(dim=1)
                train_correct += (preds == y).sum().item()
                train_total += bs
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.5f}"})
                
            self.scheduler.step()
            train_loss /= train_total
            train_acc = train_correct / train_total
            
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
                loss = self.criterion(logits[:, :10], y)
                total_loss += loss.item() * x.size(0)
                preds = logits[:, :10].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return total_loss / total, correct / total
