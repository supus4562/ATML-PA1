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
        warmup_epochs = config.get('warmup_epochs', 5 if config.get('batch_size', 128) >= 512 else 0)
        n_epochs = config['n_epochs']
        if warmup_epochs > 0 and n_epochs > warmup_epochs:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs - warmup_epochs, eta_min=1e-5
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        
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
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.5f}", 'vram': f"{vram_gb:.1f}GB"})
                else:
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
                
            vram_str = f" | Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB" if torch.cuda.is_available() else ""
            tqdm.write(
                f"[Epoch {epoch+1:03d}/{n_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} (Best: {best_acc:.4f}) | LR: {current_lr:.6f}{vram_str}"
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
