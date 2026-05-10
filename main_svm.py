import argparse
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_loader import load_victorian
from evaluate import compute_metrics, print_summary, save_results
from experiment_config import (
    AUTHOR_COL,
    CSV_PATH,
    DEV_MAX_AUTHORS,
    DEV_MAX_SAMPLES_PER_AUTHOR,
    SEED,
    TEST_SIZE,
    TEXT_COL,
)


def preprocess_text(text):
    # Houvardas 2006 trick: replace all digits with '@' to capture the 
    # stylistic habit of using numbers rather than specific dates/content.
    return re.sub(r"\d", "@", text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH, help="path to the Victorian authorship CSV")
    p.add_argument("--text-col", default=TEXT_COL)
    p.add_argument("--author-col", default=AUTHOR_COL)
    p.add_argument("--encoding", default=None, help="CSV encoding fallback.")
    p.add_argument("--max-authors", type=int, default=DEV_MAX_AUTHORS,
                   help="Number of authors to keep. Use -1 to keep all.")
    p.add_argument("--max-samples", type=int, default=DEV_MAX_SAMPLES_PER_AUTHOR,
                   help="Samples per author. Use -1 to keep all.")
    p.add_argument("--test-size", type=float, default=TEST_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default="results/svm.json")
    args = p.parse_args()

    max_samples = None if args.max_samples == -1 else args.max_samples
    max_authors = None if args.max_authors == -1 else args.max_authors

    # ---- data ----
    print("Loading data...")
    train_texts, train_y, test_texts, test_y, label_to_id = load_victorian(
        args.csv,
        text_col=args.text_col,
        author_col=args.author_col,
        encoding=args.encoding,
        max_authors=max_authors,
        max_samples_per_author=max_samples,
        test_size=args.test_size,
        seed=args.seed,
    )
    n_classes = len(label_to_id)
    print(f"  classes: {n_classes}, train: {len(train_texts)}, test: {len(test_texts)}")

    print("Applying digit masking preprocessing...")
    train_texts = [preprocess_text(t) for t in train_texts]
    test_texts = [preprocess_text(t) for t in test_texts]

    # ---- model pipeline ----
    print("Building N-gram + Feature Selection + SVM pipeline...")
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), sublinear_tf=True
                ),
            ),
            # Explicit feature selection: keeps only the top 15,000 most statistically relevant features
            ("selector", SelectKBest(chi2, k=15000)),
            ("classifier", LinearSVC(random_state=args.seed, max_iter=2000)),
        ]
    )

    # ---- train ----
    print("Training the SVM model...")
    pipeline.fit(train_texts, train_y)

    # ---- evaluate ----
    print("Evaluating on test set...")
    y_pred = pipeline.predict(test_texts)

    results = compute_metrics(test_y, y_pred, label_to_id)
    results["method"] = "N-gram + SVM"
    results["config"] = vars(args)
    results["label_to_id"] = {str(k): int(v) for k, v in label_to_id.items()}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_results(results, args.out)

    print_summary(results, "N-gram + SVM")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()