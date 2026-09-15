"""Evaluate the frozen memory write gate on a small human-authored blind set."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".models" / "memory_write_gate_memoperator_lora"
DATA = ROOT / "evals" / "memory_write_gate_blind.jsonl"
OUTPUT = ROOT / "training" / "memory_write_gate" / "metrics_blind.json"


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = json.loads((MODEL / "classifier_metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.base_model.config.pad_token_id = tokenizer.pad_token_id
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = tokenizer([row["text"] for row in rows[start:start + 16]], padding=True,
                              truncation=True, max_length=metadata["max_length"], return_tensors="pt").to("cuda")
            probabilities.extend(torch.softmax(model(**batch).logits.float(), dim=-1)[:, 1].cpu().tolist())
    truth = np.asarray([row["label_id"] for row in rows])
    predicted = (np.asarray(probabilities) >= metadata["threshold"]).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, labels=[0, 1], zero_division=0)
    report = {
        "set": "human-authored blind set",
        "rows": len(rows),
        "threshold": metadata["threshold"],
        "accuracy": float(accuracy_score(truth, predicted)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "per_label": {
            "SKIP": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])},
            "SAVE": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1])},
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "errors": [{"id": row["id"], "label": row["label"], "prediction": "SAVE" if pred else "SKIP",
                    "save_probability": round(float(prob), 6)}
                   for row, pred, prob in zip(rows, predicted, probabilities) if pred != row["label_id"]],
        "limitation": "Small author-curated diagnostic set; it is not a production benchmark.",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
