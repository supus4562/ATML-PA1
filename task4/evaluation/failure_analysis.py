import numpy as np
import pandas as pd

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Known semantic mappings for plausible confusions
PLAUSIBLE_PAIRS = {
    'bus': {'automobile', 'truck'},
    'pickup_truck': {'truck', 'automobile'},
    'motorcycle': {'automobile', 'truck', 'airplane'},
    'tractor': {'truck', 'automobile'},
    'wolf': {'dog', 'cat', 'deer'},
    'fox': {'dog', 'cat'},
    'leopard': {'cat', 'dog'},
    'camel': {'horse', 'deer'},
    'bottle': {'airplane', 'ship'},
    'bowl': {'ship'},
    'chair': set(),
    'clock': set(),
    'keyboard': set(),
    'mushroom': {'frog'},
    'sunflower': set(),
    'wardrobe': set()
}


def explain_failure(unknown_class: str, pred_cifar10: str, split: str) -> tuple[str, str]:
    """Classify failure as semantically plausible or surprising with explanation."""
    plausible_set = PLAUSIBLE_PAIRS.get(unknown_class, set())
    if pred_cifar10 in plausible_set:
        category = "Semantically Plausible"
        explanation = (
            f"Class '{unknown_class}' shares strong visual/semantic features with CIFAR-10 '{pred_cifar10}', "
            f"causing the closed-set classifier to absorb the unknown with high logit response."
        )
    else:
        category = "Surprising Failure"
        explanation = (
            f"Class '{unknown_class}' is visually distant from CIFAR-10 '{pred_cifar10}', "
            f"indicating high-confidence aliasing on background textures, spurious edges, or unconstrained open space."
        )
    return category, explanation


def find_failures(unknown_scores, unknown_labels, unknown_class_names, model_preds, threshold, split='near'):
    """Find incorrectly accepted unknown examples (u(x) <= threshold)."""
    failures = []
    # In OSR convention, u(x) <= threshold means sample is accepted as known (failure!)
    accepted_indices = np.where(unknown_scores <= threshold)[0]
    
    for idx in accepted_indices:
        score = float(unknown_scores[idx])
        margin = float(threshold - score)  # Higher margin = more confident erroneous acceptance
        unk_class = unknown_class_names[idx]
        pred_idx = int(model_preds[idx])
        pred_class = CIFAR10_CLASSES[pred_idx]
        
        category, explanation = explain_failure(unk_class, pred_class, split)
        
        failures.append({
            'index': int(idx),
            'split': split,
            'unknown_class': unk_class,
            'cifar10_predicted_class': pred_class,
            'score': round(score, 4),
            'threshold': round(float(threshold), 4),
            'margin': round(margin, 4),
            'failure_category': category,
            'explanation': explanation
        })
        
    # Sort by margin descending (most confidently accepted unknowns first)
    failures.sort(key=lambda x: x['margin'], reverse=True)
    return failures


def compute_per_class_rejection(unknown_scores, unknown_class_names, model_preds, threshold, split='near'):
    """Compute rejection statistics per fine unknown class."""
    df = pd.DataFrame({
        'unknown_class': unknown_class_names,
        'score': unknown_scores,
        'pred_idx': model_preds,
    })
    df['cifar10_pred'] = df['pred_idx'].apply(lambda i: CIFAR10_CLASSES[i])
    df['rejected'] = df['score'] > threshold  # Correct rejection when score > threshold
    
    records = []
    for cls_name, grp in df.groupby('unknown_class'):
        total = len(grp)
        n_rejected = int(grp['rejected'].sum())
        n_accepted = total - n_rejected
        rej_rate = n_rejected / total
        top_confusion = grp[~grp['rejected']]['cifar10_pred'].mode()
        top_pred_str = top_confusion.iloc[0] if len(top_confusion) > 0 else "None (100% rejected)"
        
        records.append({
            'split': split,
            'unknown_class': cls_name,
            'total_samples': total,
            'accepted_count': n_accepted,
            'rejected_count': n_rejected,
            'rejection_rate': round(rej_rate * 100.0, 2),
            'fpr_at_95tpr': round((n_accepted / total) * 100.0, 2),
            'top_cifar10_confusion': top_pred_str
        })
        
    return records
