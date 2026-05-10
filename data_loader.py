"""Shared data loading for both methods.

The original Victorian dataset (Gungor 2018) has 45 authors in training and
50 in test (5 unknown). The Kaggle CSV typically contains everything in a
single file with a numeric author label, so here we do a stratified
train/test split ourselves. This keeps the comparison between BertAA and
the n-gram+SVM method identical.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_with_fallback(csv_path: str, encoding: str | None = None) -> pd.DataFrame:
    """Read a CSV, trying encodings used by the Victorian dataset if needed."""
    encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"could not read {csv_path} with encodings: {encodings}",
    )


def load_victorian(
    csv_path: str,
    text_col: str = "text",
    author_col: str = "author",
    encoding: str | None = None,
    max_authors: int | None = None,
    max_samples_per_author: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Load the Victorian authorship dataset and produce a train/test split.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    text_col, author_col : str
        Column names in the CSV.
    encoding : str, optional
        CSV encoding. If omitted, tries UTF-8 first and falls back to cp1252
        and latin1, which handles the Gungor Victorian CSV in this project.
    max_authors : int, optional
        If set, keep only this many authors (smallest IDs first). Useful for
        CPU experiments.
    max_samples_per_author : int, optional
        If set, downsample each author to this many examples.
    test_size : float
        Fraction of the data to hold out. Stratified by author so each
        author appears in both splits.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_texts, train_labels, test_texts, test_labels, label_to_id
        Lists/arrays for training and testing. ``label_to_id`` maps the
        original author label to a contiguous integer class index (0..K-1).
    """
    df = read_csv_with_fallback(csv_path, encoding=encoding)
    if text_col not in df.columns or author_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{text_col}' and '{author_col}'. "
            f"Found: {list(df.columns)}"
        )

    df = df[[text_col, author_col]].dropna()
    df[text_col] = df[text_col].astype(str)

    # Subset authors (helpful on CPU)
    if max_authors is not None:
        kept = sorted(df[author_col].unique())[:max_authors]
        df = df[df[author_col].isin(kept)]

    # Subset samples per author. Note: we cannot pass ``n`` larger than the
    # smallest group, so cap by group size first then sample uniformly.
    if max_samples_per_author is not None:
        parts = []
        for _, g in df.groupby(author_col, group_keys=False):
            n = min(len(g), max_samples_per_author)
            parts.append(g.sample(n=n, random_state=seed))
        df = pd.concat(parts, ignore_index=True)

    # Map labels to contiguous ints starting at 0
    unique_authors = sorted(df[author_col].unique())
    label_to_id = {a: i for i, a in enumerate(unique_authors)}
    df["label"] = df[author_col].map(label_to_id)

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )

    return (
        train_df[text_col].tolist(),
        train_df["label"].to_numpy(),
        test_df[text_col].tolist(),
        test_df["label"].to_numpy(),
        label_to_id,
    )
