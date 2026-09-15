"""Fine-tune official multilingual DistilBERT for scope-intent classification."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "training" / "scope_intent" / "scope_intent_20000.jsonl"
MODEL_DIR = ROOT / ".models" / "scope_intent_distilmbert"
REPORT_PATH = ROOT / "training" / "scope_intent" / "metrics.json"
BASE_MODEL = "distilbert/distilbert-base-multilingual-cased"


class Rows(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"], truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None,
        )
        encoded["labels"] = int(row["label_id"])
        return encoded


def collator(tokenizer):
    def collate(items):
        labels = torch.tensor([item.pop("labels") for item in items], dtype=torch.long)
        batch = tokenizer.pad(items, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch
    return collate


def probabilities(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            y = batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(y.tolist())
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def metrics(y_true, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    result = {}
    for label_id, name in ((0, "DIRECT_RESPONSE"), (1, "REQUIRES_SCOPE_CONTRACT")):
        tp = int(((predicted == label_id) & (y_true == label_id)).sum())
        fp = int(((predicted == label_id) & (y_true != label_id)).sum())
        fn = int(((predicted != label_id) & (y_true == label_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {"precision": precision, "recall": recall, "f1": f1}
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    order = np.argsort(probability)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(probability))
    pos = ranks[y_true == 1]
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    auc = (float(pos.sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {
        "threshold": float(threshold), "accuracy": float((predicted == y_true).mean()),
        "roc_auc": auc, "per_label": result, "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def choose_threshold(y_true, probability, minimum_recall=0.98):
    best = None
    for threshold in np.unique(np.round(probability, 6)):
        predicted = probability >= threshold
        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        if recall >= minimum_recall:
            candidate = (fp, -float(threshold), float(threshold))
            best = candidate if best is None or candidate < best else best
    return best[2] if best else 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    random.seed(20260913)
    np.random.seed(20260913)
    torch.manual_seed(20260913)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(row["label_id"] for row in rows)
    if not rows or counts[0] != counts[1] or set(counts) != {0, 1}:
        raise SystemExit("expected a non-empty balanced binary dataset")
    by_split = {split: [row for row in rows if row["split"] == split]
                for split in ("train", "validation", "test")}
    split_seeds = {split: {row["seed_id"] for row in values} for split, values in by_split.items()}
    if any(split_seeds[a] & split_seeds[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise SystemExit("seed leakage detected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2,
        id2label={0: "DIRECT_RESPONSE", 1: "REQUIRES_SCOPE_CONTRACT"},
        label2id={"DIRECT_RESPONSE": 0, "REQUIRES_SCOPE_CONTRACT": 1},
    ).to(device)
    loaders = {
        split: DataLoader(Rows(values, tokenizer, args.max_length),
                          batch_size=args.batch_size, shuffle=split == "train",
                          collate_fn=collator(tokenizer), num_workers=0)
        for split, values in by_split.items()
    }
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(loaders["train"], 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**{k: v.to(device) for k, v in batch.items()})
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
        y_val, p_val = probabilities(model, loaders["validation"], device)
        threshold = choose_threshold(y_val, p_val)
        epoch_metrics = metrics(y_val, p_val, threshold)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses),
                        "validation": epoch_metrics})
        print(json.dumps(history[-1], ensure_ascii=False))

    y_val, p_val = probabilities(model, loaders["validation"], device)
    threshold = choose_threshold(y_val, p_val)
    y_test, p_test = probabilities(model, loaders["test"], device)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL, "developer": "Hugging Face DistilBERT team",
        "classifier": "AutoModelForSequenceClassification", "encoder_frozen": False,
        "device": str(device), "epochs": args.epochs, "batch_size": args.batch_size,
        "rows": {key: len(value) for key, value in by_split.items()},
        "seed_counts": {key: len(value) for key, value in split_seeds.items()},
        "threshold_selection": "minimize validation false positives subject to positive recall >= 0.98",
        "history": history, "validation": metrics(y_val, p_val, threshold),
        "test": metrics(y_test, p_test, threshold),
        "limitation": "Synthetic held-out seed families measure generator-domain generalization, not production accuracy.",
    }
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.model_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.model_dir)
    (args.model_dir / "router_metadata.json").write_text(
        json.dumps({"threshold": threshold, "labels": {"0": "DIRECT_RESPONSE",
        "1": "REQUIRES_SCOPE_CONTRACT"}, "max_length": args.max_length},
        ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
