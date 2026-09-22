import torch
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead
from common.metrics import calculate_accuracy, calculate_macro_f1

class ERMBaseline:
    def __init__(self, checkpoint_path, device):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.backbone = ResNet18Backbone().to(device)
        self.classifier = ClassifierHead().to(device)
        self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        bb_sd = checkpoint['backbone_state_dict']
        if any(k.startswith("features.conv1.") for k in bb_sd.keys()):
            mapping = {
                "features.conv1.": "features.0.",
                "features.bn1.": "features.1.",
                "features.layer1.": "features.4.",
                "features.layer2.": "features.5.",
                "features.layer3.": "features.6.",
                "features.layer4.": "features.7.",
            }
            new_sd = {}
            for k, v in bb_sd.items():
                new_k = k
                for old_p, new_p in mapping.items():
                    if k.startswith(old_p):
                        new_k = new_p + k[len(old_p):]
                        break
                new_sd[new_k] = v
            bb_sd = new_sd

        self.backbone.load_state_dict(bb_sd)
        self.classifier.load_state_dict(checkpoint['head_state_dict'])


    def evaluate(self, val_loaders):
        self.backbone.eval()
        self.classifier.eval()
        
        results = {}
        with torch.no_grad():
            for domain_name, loader in val_loaders.items():
                all_preds, all_labels = [], []
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.classifier(self.backbone(x))
                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(y.cpu())
                
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                results[domain_name] = {
                    'accuracy': calculate_accuracy(all_preds.numpy(), all_labels.numpy()),
                    'macro_f1': calculate_macro_f1(all_preds.numpy(), all_labels.numpy())
                }
        return results
