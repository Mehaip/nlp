"""BertAA: BERT fine-tuning for authorship attribution.

Implements the architecture from Fabien et al. (2020):

    Input text -> BERT -> [CLS] hidden state -> Dense -> Softmax -> author

The defaults follow the paper's setup as closely as possible for this project:
``bert-base-cased`` and a 512-token input window. For a faster CPU smoke test,
pass a smaller subset, for example:

    --max-authors 5 --max-samples 20 --max-length 128 --epochs 1
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from data_loader import load_victorian
from evaluate import compute_metrics, save_results, print_summary
from experiment_config import (
    AUTHOR_COL,
    CSV_PATH,
    DEV_MAX_AUTHORS,
    DEV_MAX_SAMPLES_PER_AUTHOR,
    SEED,
    TEST_SIZE,
    TEXT_COL,
)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class BertAA(nn.Module):
    """BERT + dropout + dense + softmax classifier (the paper's architecture).

    The softmax is applied implicitly by ``nn.CrossEntropyLoss`` during
    training, so ``forward`` returns raw logits. Equivalent at inference
    time since argmax(softmax(x)) == argmax(x).
    """

    def __init__(self, model_name: str, num_authors: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_authors)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] is the first token; its hidden state summarises the whole input.
        cls_hidden = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_hidden))


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }


# ----------------------------------------------------------------------------
# Training / evaluation loops
# ----------------------------------------------------------------------------

def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)
        optim.zero_grad()
        logits = model(ids, mask)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        all_pred.extend(logits.argmax(dim=-1).cpu().numpy())
        all_true.extend(batch["label"].numpy())
    return np.array(all_true), np.array(all_pred)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default=CSV_PATH,
        help="path to the Victorian authorship CSV",
    )
    p.add_argument("--text-col", default=TEXT_COL)
    p.add_argument("--author-col", default=AUTHOR_COL)
    p.add_argument("--encoding", default=None,
                   help="CSV encoding. Default tries utf-8, cp1252, then latin1.")
    p.add_argument("--model", default="bert-base-cased",
                   help="HF model name. BertAA paper uses bert-base-cased.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-authors", type=int, default=DEV_MAX_AUTHORS,
                   help="Number of authors to keep. Use -1 to keep all.")
    p.add_argument("--max-samples", type=int, default=DEV_MAX_SAMPLES_PER_AUTHOR,
                   help="Samples per author. Use -1 to keep all.")
    p.add_argument("--test-size", type=float, default=TEST_SIZE,
                   help="Held-out test fraction. Keep identical across methods.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default="results/bertaa.json")
    p.add_argument("--save-model", default=None,
                   help="Optional directory for the fine-tuned model and tokenizer.")
    args = p.parse_args()

    max_samples = None if args.max_samples == -1 else args.max_samples
    max_authors = None if args.max_authors == -1 else args.max_authors

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    # ---- model ----
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertAA(args.model, num_authors=n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model.to(device)

    train_ds = TextDataset(train_texts, train_y, tokenizer, args.max_length)
    test_ds = TextDataset(test_texts, test_y, tokenizer, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    # ---- train ----
    optim = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_dl, optim, loss_fn, device)
        dt = time.time() - t0
        print(f"epoch {epoch}: loss={loss:.4f}  ({dt:.1f}s)")

    # ---- evaluate ----
    print("Evaluating on test set...")
    y_true, y_pred = predict(model, test_dl, device)

    results = compute_metrics(y_true, y_pred, label_to_id)
    results["method"] = "BertAA"
    results["config"] = vars(args)
    results["label_to_id"] = {str(k): int(v) for k, v in label_to_id.items()}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_results(results, args.out)

    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        model.bert.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        torch.save(model.classifier.state_dict(), os.path.join(args.save_model, "classifier.pt"))

    print_summary(results, "BertAA")
    print(f"Saved to {args.out}")
    if args.save_model:
        print(f"Saved model to {args.save_model}")


if __name__ == "__main__":
    main()
