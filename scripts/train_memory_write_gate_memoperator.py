"""LoRA fine-tune MemOperator-0.6B with a deterministic SAVE/SKIP head."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "training" / "memory_write_gate" / "memory_write_gate_20000.jsonl"
MODEL_DIR = ROOT / ".models" / "memory_write_gate_memoperator_lora"
REPORT_PATH = ROOT / "training" / "memory_write_gate" / "metrics_memoperator_lora.json"
BASE_MODEL = "MemTensor/MemOperator-0.6B"
LABELS = {0: "SKIP", 1: "SAVE"}


class Rows(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
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
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**inputs).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
            labels.extend(y.tolist())
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def metrics(y_true, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    per_label = {}
    for label_id, name in LABELS.items():
        tp = int(((predicted == label_id) & (y_true == label_id)).sum())
        fp = int(((predicted == label_id) & (y_true != label_id)).sum())
        fn = int(((predicted != label_id) & (y_true == label_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[name] = {"precision": precision, "recall": recall, "f1": f1}
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    order = np.argsort(probability)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(probability))
    positives = ranks[y_true == 1]
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    auc = (float(positives.sum()) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return {
        "threshold": float(threshold),
        "accuracy": float((predicted == y_true).mean()),
        "roc_auc": auc,
        "per_label": per_label,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def choose_threshold(y_true, probability, minimum_save_recall: float):
    best = None
    for threshold in np.unique(np.round(probability, 6)):
        predicted = probability >= threshold
        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        if recall >= minimum_save_recall:
            candidate = (fp, -float(threshold), float(threshold))
            best = candidate if best is None or candidate < best else best
    return best[2] if best else 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-save-recall", type=float, default=0.97)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    args = parser.parse_args()

    random.seed(20260915)
    np.random.seed(20260915)
    torch.manual_seed(20260915)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(row["label_id"] for row in rows)
    if not rows or counts[0] != counts[1] or set(counts) != {0, 1}:
        raise SystemExit("expected a non-empty balanced binary dataset")
    by_split = {split: [row for row in rows if row["split"] == split]
                for split in ("train", "validation", "test")}
    split_seeds = {split: {row["seed_id"] for row in values} for split, values in by_split.items()}
    if any(split_seeds[a] & split_seeds[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise SystemExit("seed-family leakage detected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=LABELS,
        label2id={name: index for index, name in LABELS.items()},
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora).to(device)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())

    loaders = {
        split: DataLoader(
            Rows(values, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=split == "train",
            collate_fn=collator(tokenizer),
            num_workers=0,
        )
        for split, values in by_split.items()
    }
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    history = []
    best_state = None
    best_epoch = 0
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loaders["train"], 1):
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**inputs).loss / args.gradient_accumulation
            loss.backward()
            if step % args.gradient_accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().float().cpu()) * args.gradient_accumulation)
        y_val, p_val = probabilities(model, loaders["validation"], device)
        threshold = choose_threshold(y_val, p_val, args.minimum_save_recall)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses),
               "validation": metrics(y_val, p_val, threshold)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        accuracy = float(row["validation"]["accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(json.dumps({"early_stop": True, "best_epoch": best_epoch,
                                  "best_validation_accuracy": best_accuracy}), flush=True)
                break

    if best_state is None:
        raise RuntimeError("training did not produce a best checkpoint")
    model.load_state_dict(best_state, strict=False)

    y_val, p_val = probabilities(model, loaders["validation"], device)
    threshold = choose_threshold(y_val, p_val, args.minimum_save_recall)
    y_test, p_test = probabilities(model, loaders["test"], device)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "architecture": "Qwen3ForSequenceClassification + PEFT LoRA",
        "device": str(device), "dtype": str(dtype),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "rows": {key: len(value) for key, value in by_split.items()},
        "seed_counts": {key: len(value) for key, value in split_seeds.items()},
        "threshold_selection": f"minimize validation false positives subject to SAVE recall >= {args.minimum_save_recall}",
        "history": history,
        "validation": metrics(y_val, p_val, threshold),
        "test": metrics(y_test, p_test, threshold),
        "limitation": "Synthetic held-out seed families do not replace a human-labeled production blind set.",
    }
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.model_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.model_dir)
    (args.model_dir / "classifier_metadata.json").write_text(
        json.dumps({"threshold": threshold, "labels": LABELS, "max_length": args.max_length,
                    "base_model": args.base_model}, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
