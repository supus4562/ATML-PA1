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
        num_classes = len(config.get("classes", range(10)))
        self.head = nn.Linear(feat_dim, num_classes).to(self.device)
        
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=config.get('lr', 1e-3), weight_decay=config.get('weight_decay', 1e-4))
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        patience = config.get('patience', 5)
        no_improve = 0
        
        for epoch in range(config.get('max_epochs', 50)):
            self.head.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    if self.name == "ResNet-50":
                        feats = self.model(imgs)
                    elif self.name == "ViT-B/16":
                        self.features = []
                        self.model(imgs)
                        feats = self.features[0]
                    elif self.name == "CLIP":
                        feats = self.model.encode_image(imgs)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                
                preds = self.head(feats)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Eval
            self.head.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    if self.name == "ResNet-50":
                        feats = self.model(imgs)
                    elif self.name == "ViT-B/16":
                        self.features = []
                        self.model(imgs)
                        feats = self.features[0]
                    elif self.name == "CLIP":
                        feats = self.model.encode_image(imgs)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        
                    preds = self.head(feats)
                    correct += (preds.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            
            acc = correct / total
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
