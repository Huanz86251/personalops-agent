"""Inference wrapper for the fine-tuned multilingual DistilBERT scope router."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / ".models" / "scope_intent_distilmbert"


class ScopeIntentClassifier:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        metadata = json.loads((self.model_dir / "router_metadata.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.threshold = float(metadata["threshold"])
        self.max_length = int(metadata["max_length"])

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=self.max_length)
        with torch.inference_mode():
            probability = float(torch.softmax(self.model(**inputs).logits, dim=-1)[0, 1])
        requires_scope = probability >= self.threshold
        return {
            "label": "REQUIRES_SCOPE_CONTRACT" if requires_scope else "DIRECT_RESPONSE",
            "label_id": int(requires_scope),
            "probability_requires_scope": probability, "threshold": self.threshold,
        }
