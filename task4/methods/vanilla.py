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
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(self.config['n_epochs']):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['n_epochs']}")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * x.size(0)
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
        
    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
