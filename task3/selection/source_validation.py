import json
import os

def select_best_checkpoint(results_dir, method_name):
    history_file = os.path.join(results_dir, f"{method_name}_training_history.json")
    if not os.path.exists(history_file):
        return None
    return os.path.join(results_dir, f"{method_name}_checkpoint.pth")
