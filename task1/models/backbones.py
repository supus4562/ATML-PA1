from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import open_clip
import numpy as np

MODEL_NAMES = ["ResNet-50", "ViT-B/16", "CLIP"]

class BackboneExtractor:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        
        if name == "ResNet-50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Identity()
            self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        elif name == "ViT-B/16":
            self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.model.heads = nn.Identity()
            self.transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            # Hook class token
            self.features = []
            def hook(m, i, o):
                self.features.append(o[:, 0])
            self.model.encoder.ln.register_forward_hook(hook)
        elif name == "CLIP":
            self.model, _, self.transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        
        self.model.eval().to(device)
        self.head = None

    def get_transform(self):
        return self.transform

    def extract_features(self, dataloader):
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                if self.name == "ResNet-50":
                    feats = self.model(imgs)
                elif self.name == "ViT-B/16":
                    self.features = []
                    self.model(imgs)
                    feats = self.features[0]
                elif self.name == "CLIP":
                    feats = self.model.encode_image(imgs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    
                all_feats.append(feats.cpu().numpy())
                if labels is not None:
                    all_labels.append(labels.cpu().numpy())
        return np.concatenate(all_feats), (np.concatenate(all_labels) if len(all_labels) > 0 else None)

    def train_linear_head(self, train_loader, val_loader, config):
        feat_dim = 2048 if self.name == "ResNet-50" else (768 if self.name == "ViT-B/16" else 512)
        num_classes = len(config.get("classes", range(37)))
        self.head = nn.Linear(feat_dim, num_classes).to(self.device)
        
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=config.get('lr', 1e-3), weight_decay=config.get('weight_decay', 1e-4))
        criterion = nn.CrossEntropyLoss()
        
        # PRE-EXTRACT FEATURES ONCE TO GPU (MASSIVE SPEEDUP)
        train_feats, train_labels = self.extract_features(train_loader)
        val_feats, val_labels = self.extract_features(val_loader)
        
        train_feats = torch.tensor(train_feats, device=self.device)
        train_labels = torch.tensor(train_labels, dtype=torch.long, device=self.device)
        val_feats = torch.tensor(val_feats, device=self.device)
        val_labels = torch.tensor(val_labels, dtype=torch.long, device=self.device)
        
        batch_size = config.get("batch_size", 1024)
        dataset_size = train_feats.shape[0]
        
        best_acc = 0
        patience = config.get('patience', 5)
        no_improve = 0
        
        for epoch in tqdm(range(config.get('max_epochs', 50)), desc=f'Training {self.name} head'):
            self.head.train()
            perm = torch.randperm(dataset_size, device=self.device)
            for i in range(0, dataset_size, batch_size):
                idx = perm[i:i+batch_size]
                feats = train_feats[idx]
                labels = train_labels[idx]
                
                preds = self.head(feats)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Eval
            self.head.eval()
            with torch.no_grad():
                val_preds = self.head(val_feats)
                acc = (val_preds.argmax(1) == val_labels).float().mean().item()
            
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        return {"best_val_acc": best_acc}

    def predict(self, dataloader):
        self.head.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                if self.name == "ResNet-50":
                    feats = self.model(imgs)
                elif self.name == "ViT-B/16":
                    self.features = []
                    self.model(imgs)
                    feats = self.features[0]
                elif self.name == "CLIP":
                    feats = self.model.encode_image(imgs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    
                logits = self.head(feats)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(logits.argmax(-1).cpu().numpy())
                if labels is not None:
                    all_labels.append(labels.cpu().numpy())
                    
        return np.concatenate(all_preds), np.concatenate(all_probs), (np.concatenate(all_labels) if len(all_labels) > 0 else None)


class CLIPZeroShot:
    def __init__(self, device):
        self.device = device
        self.model, _, self.transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def predict(self, dataloader, class_names):
        text_inputs = torch.cat([self.tokenizer(f"a photo of a {c}") for c in class_names]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            all_preds = []
            all_probs = []
            all_labels = []
            
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)
                image_features = self.model.encode_image(imgs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                all_probs.append(similarity.cpu().numpy())
                all_preds.append(similarity.argmax(dim=-1).cpu().numpy())
                if labels is not None:
                    all_labels.append(labels.cpu().numpy())
                    
        return np.concatenate(all_preds), np.concatenate(all_probs), (np.concatenate(all_labels) if len(all_labels) > 0 else None)
