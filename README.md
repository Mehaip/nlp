# Victorian Authorship Attribution with BertAA

This project implements the BERT method from the report:

```text
Input text -> BERT -> [CLS] hidden state -> Dense layer -> author logits
```

The training data is `Gungor_2018_VictorianAuthorAttribution_data-train.csv`,
with `text` as the input column and `author` as the label column.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Shared Comparison Protocol

Both methods should use the same dataset loader and evaluator:

- dataset: `Gungor_2018_VictorianAuthorAttribution_data-train.csv`
- input column: `text`
- label column: `author`
- split: `80/20`, stratified by author
- random seed: `42`
- development subset: `10` authors x `100` samples per author
- final subset: `25` authors x `500` samples per author
- output metrics: accuracy, macro-F1, weighted-F1, per-author metrics, confusion matrix

The shared constants are in `experiment_config.py`. The BERT script imports
them, and the other method should do the same or use the same values.

## Development Run

This is the agreed common development setup.

```bash
python train_bertaa.py \
  --max-authors 10 \
  --max-samples 100 \
  --test-size 0.2 \
  --seed 42 \
  --out results/bertaa_dev.json
```

## Full BertAA-style run

This keeps the same comparison protocol but uses the agreed final subset. It is
meant for a GPU.

```bash
python train_bertaa.py \
  --model bert-base-cased \
  --max-length 512 \
  --max-authors 25 \
  --max-samples 500 \
  --test-size 0.2 \
  --seed 42 \
  --batch-size 16 \
  --epochs 5 \
  --out results/bertaa_full.json \
  --save-model models/bertaa_victorian
```

The script writes accuracy, macro-F1, weighted-F1, per-author metrics, the
confusion matrix, and the author label mapping to the JSON output file.
