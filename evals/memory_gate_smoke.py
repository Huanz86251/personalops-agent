"""Smoke benchmark for candidate lightweight multilingual memory gates.

This is an offline diagnostic, not a production threshold benchmark.  It
measures warm local inference and exposes the raw NLI probabilities so that
model routing is not chosen from parameter counts alone.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class Case:
    language: str
    text: str
    expected: str


CASES = (
    Case("zh", "以后回答我时默认使用中文。", "save"),
    Case("zh", "我对花生严重过敏。", "save"),
    Case("zh", "我最近可能要搬到上海，但还没有确定。", "gray"),
    Case("zh", "请把这段文字翻译成英文。", "drop"),
    Case("zh", "谢谢，继续吧。", "drop"),
    Case("en", "Please answer me concisely in future conversations.", "save"),
    Case("en", "I am severely allergic to peanuts.", "save"),
    Case("en", "Translate this paragraph into Chinese.", "drop"),
    Case("en", "Thanks, keep going.", "drop"),
    Case("mixed", "以后回答 technical questions 时 please use English terms。", "save"),
)

HYPOTHESES = {
    "zh": (
        "这段话陈述了用户本人的个人事实、长期偏好、习惯、目标或约束。",
        "这段话是对助手的命令、问题、寒暄或一次性任务。",
    ),
    "en": (
        "This message states the user's personal fact, long-term preference, habit, goal, or constraint.",
        "This message is a command, question, greeting, or one-time task for the assistant.",
    ),
}

MODELS = (
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    "IDEA-CCNL/Erlangshen-Roberta-110M-NLI",
)


def _label_index(model, label: str) -> int:
    normalized = label.casefold()
    for index, name in model.config.id2label.items():
        if str(name).casefold() == normalized:
            return int(index)
    raise RuntimeError(f"Model does not expose an {label!r} NLI label")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = r"D:\PythonProject\.models"
    print(f"device={device}")

    for model_name in MODELS:
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        ).to(device)
        model.eval()
        load_seconds = time.perf_counter() - started
        entailment_index = _label_index(model, "entailment")
        contradiction_index = _label_index(model, "contradiction")

        # The Chinese-only model is deliberately evaluated only on Chinese;
        # mixed input belongs to the multilingual fallback.
        cases = CASES if "multilingual" in model_name else tuple(
            case for case in CASES if case.language == "zh"
        )
        pairs: list[tuple[str, str]] = []
        for case in cases:
            hypothesis_language = "zh" if case.language == "zh" else "en"
            for hypothesis in HYPOTHESES[hypothesis_language]:
                pairs.append((case.text, hypothesis))

        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            model(**encoded)
        if device == "cuda":
            torch.cuda.synchronize()

        timings_ms: list[float] = []
        probabilities = None
        for _ in range(10):
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
            if device == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - started) * 1000)

        assert probabilities is not None

        single_pair_batch = pairs[:2]
        end_to_end_ms: list[float] = []
        for _ in range(30):
            started = time.perf_counter()
            single_encoded = tokenizer(
                single_pair_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                model(**single_encoded)
            if device == "cuda":
                torch.cuda.synchronize()
            end_to_end_ms.append((time.perf_counter() - started) * 1000)

        # Hugging Face zero-shot multi-label classification normalizes the
        # entailment and contradiction logits for each candidate label.  Raw
        # three-way entailment probabilities are not calibrated label scores.
        binary = probabilities[:, [contradiction_index, entailment_index]]
        zero_shot_scores = (
            binary[:, 1] / binary.sum(dim=-1)
        ).detach().cpu().tolist()
        print(f"\nmodel={model_name}")
        print(f"load_seconds={load_seconds:.3f}")
        print(
            "batch_ms="
            f"median:{statistics.median(timings_ms):.3f} "
            f"p95:{sorted(timings_ms)[-1]:.3f} "
            f"pairs:{len(pairs)}"
        )
        print(
            "single_message_e2e_ms="
            f"median:{statistics.median(end_to_end_ms):.3f} "
            f"p95:{sorted(end_to_end_ms)[-2]:.3f} pairs:2"
        )
        for case_index, case in enumerate(cases):
            durable = zero_shot_scores[case_index * 2]
            temporary = zero_shot_scores[case_index * 2 + 1]
            print(
                f"expected={case.expected:4s} "
                f"durable={durable:.4f} temporary={temporary:.4f} "
                f"text={case.text}"
            )


if __name__ == "__main__":
    main()
