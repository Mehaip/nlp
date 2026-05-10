"""Shared evaluation: accuracy, macro-F1, per-class F1, confusion matrix.

Both methods call into this module so the comparison is exactly identical.
Results are saved as JSON for ``compare.py`` to read.
"""

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, label_to_id: dict) -> dict:
    """Compute the standard authorship-attribution metrics."""
    id_to_label = {i: str(a) for a, i in label_to_id.items()}
    labels_sorted = sorted(id_to_label.keys())
    target_names = [id_to_label[i] for i in labels_sorted]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = classification_report(
        y_true, y_pred,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels_sorted
        ).tolist(),
        "n_classes": len(labels_sorted),
        "n_test_samples": int(len(y_true)),
    }


def save_results(results: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_summary(results: dict, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro F1:    {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    print(f"  Classes:     {results['n_classes']}")
    print(f"  Test size:   {results['n_test_samples']}")
