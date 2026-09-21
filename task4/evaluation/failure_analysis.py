import numpy as np

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def find_failures(unknown_scores, unknown_labels, unknown_class_names, model_preds, threshold, split='near'):
    failures = []
    accepted = np.where(unknown_scores < threshold)[0]
    
    for idx in accepted:
        failures.append({
            'unknown_class': unknown_class_names[idx],
            'cifar10_predicted_class': CIFAR10_CLASSES[model_preds[idx]],
            'score': float(unknown_scores[idx]),
            'threshold': float(threshold)
        })
    return failures
