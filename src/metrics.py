"""Shared evaluation helpers. Every model is scored on the SAME test.csv with these functions."""
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

RESULTS_PATH = "results/test_metrics.json"


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def best_threshold(y_true, proba, grid=np.arange(0.05, 0.96, 0.05)):
    """Pick the decision threshold with the highest macro-F1. Call ONLY with validation data."""
    scores = [(float(t), macro_f1(y_true, (np.asarray(proba) >= t).astype(int))) for t in grid]
    return max(scores, key=lambda x: x[1])


def evaluate(y_true, y_pred) -> dict:
    return {
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_offensive": round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
        "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=["clean", "offensive"],
                                        digits=4, output_dict=True),
    }


def save_test_result(model_name: str, result: dict, path: str = RESULTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[model_name] = result
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
