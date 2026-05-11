import argparse
import json


def load_results(path: str) -> dict:
    """Loads evaluation metrics from a specified JSON filepath."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(
        description="Performs a formal comparative analysis of evaluation metrics between classification models."
    )
    p.add_argument(
        "--svm", default="results/svm.json", help="Filepath to the traditional ML (SVM) results JSON."
    )
    p.add_argument(
        "--bertaa", default="results/bertaa_full.json", help="Filepath to the deep learning (BertAA) results JSON."
    )
    args = p.parse_args()

    try:
        svm_res = load_results(args.svm)
    except FileNotFoundError:
        print(f"Error: Unable to locate {args.svm}. Please ensure the SVM evaluation pipeline has completed.")
        return

    try:
        bert_res = load_results(args.bertaa)
    except FileNotFoundError:
        print(f"Error: Unable to locate {args.bertaa}. Please ensure the BertAA evaluation pipeline has completed.")
        return

    svm_name = svm_res.get("method", "N-gram + SVM")
    bert_name = bert_res.get("method", "BertAA")

    # ---- 1. Global Metrics Analysis ----
    print("\nSection 1: Global Performance Evaluation")
    print("-" * 75)
    print(f"{'Evaluation Metric':<20} | {svm_name:<15} | {bert_name:<15} | {'Variance (SVM - BERT)':<20}")
    print("-" * 75)

    metrics = [
        ("Overall Accuracy", "accuracy"),
        ("Macro-Averaged F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    ]

    for display_name, key in metrics:
        svm_val = svm_res[key]
        bert_val = bert_res[key]
        variance = svm_val - bert_val
        print(f"{display_name:<20} | {svm_val:.4f}{' ':<9} | {bert_val:.4f}{' ':<9} | {variance:+.4f}")

    print("-" * 75)
    print(f"Evaluation Context: {svm_res['n_test_samples']} test samples evaluated across {svm_res['n_classes']} distinct author classes.")

    # ---- 2. Granular Class-Level Breakdown ----
    print("\nSection 2: Granular Class-Level Analysis (F1-Score)")
    print("-" * 75)

    svm_classes = svm_res["per_class"]
    bert_classes = bert_res["per_class"]

    # Sort class labels numerically for structured reading
    class_labels = sorted(svm_classes.keys(), key=lambda x: int(x) if x.isdigit() else x)

    svm_superiority_count = 0
    bert_superiority_count = 0
    equilibrium_count = 0

    for label in class_labels:
        if label not in bert_classes:
            continue

        svm_f1 = svm_classes[label]["f1"]
        bert_f1 = bert_classes[label]["f1"]

        if svm_f1 > bert_f1 + 0.0001:
            optimal_model = f"{svm_name} (+{svm_f1 - bert_f1:.4f})"
            svm_superiority_count += 1
        elif bert_f1 > svm_f1 + 0.0001:
            optimal_model = f"{bert_name} (+{bert_f1 - svm_f1:.4f})"
            bert_superiority_count += 1
        else:
            optimal_model = "Equilibrium (Comparable Performance)"
            equilibrium_count += 1

        print(f"Class Identifier {label:<4} | SVM: {svm_f1:.4f} | BERT: {bert_f1:.4f} | Superior Model: {optimal_model}")

    print("-" * 75)
    print("Aggregate Distribution of Superior Performance:")
    print(f"  {svm_name}: Superior in {svm_superiority_count} classes")
    print(f"  {bert_name}: Superior in {bert_superiority_count} classes")
    print(f"  Comparable Performance Observed: {equilibrium_count} classes")
    print("-" * 75 + "\n")


if __name__ == "__main__":
    main()