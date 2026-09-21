def get_mean_metrics(results):
    acc = sum(r['accuracy'] for r in results.values()) / len(results)
    f1 = sum(r['macro_f1'] for r in results.values()) / len(results)
    return {'accuracy': acc, 'macro_f1': f1}

def get_worst_metrics(results):
    worst_acc = min(r['accuracy'] for r in results.values())
    worst_f1 = min(r['macro_f1'] for r in results.values())
    return {'accuracy': worst_acc, 'macro_f1': worst_f1}
