"""Local two-stage prompt-injection filtering for tool-returned text.

The guard deliberately operates on the text that would be shown to the model.
It preserves LangChain/MCP result envelopes and replaces only confirmed spans.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

DEFAULT_PRIMARY_MODEL = "patronus-studio/wolf-defender-prompt-injection-small"
DEFAULT_PRIMARY_ONNX = "onnx/int8_int4_embeddings/model.onnx"
DEFAULT_SECONDARY_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"

_STRUCTURAL_STRING_KEYS = frozenset(
    {
        "url",
        "source_url",
        "final_url",
        "path",
        "filename",
        "candidate_id",
        "mime",
        "content_type",
        "status",
        "hash",
        "sha256",
        "tool_call_id",
        "type",
    }
)


@dataclass(frozen=True)
class PromptInjectionGuardConfig:
    enabled: bool = True
    cache_dir: Path = Path(".models")
    primary_model: str = DEFAULT_PRIMARY_MODEL
    primary_onnx_file: str = DEFAULT_PRIMARY_ONNX
    primary_threshold: float = 0.5
    primary_window_tokens: int = 2048
    primary_overlap_tokens: int = 64
    primary_batch_size: int = 8
    secondary_model: str = DEFAULT_SECONDARY_MODEL
    secondary_window_tokens: int = 512
    secondary_overlap_tokens: int = 40
    secondary_batch_size: int = 4
    secondary_device: str = "auto"
    cache_entries: int = 2048

    def __post_init__(self) -> None:
        if not 0.0 <= self.primary_threshold <= 1.0:
            raise ValueError("primary_threshold must be between 0 and 1")
        if not 0 <= self.primary_overlap_tokens < self.primary_window_tokens:
            raise ValueError("primary overlap must be smaller than its window")
        if not 0 <= self.secondary_overlap_tokens < self.secondary_window_tokens:
            raise ValueError("secondary overlap must be smaller than its window")
        if self.primary_batch_size < 1 or self.secondary_batch_size < 1:
            raise ValueError("guard batch sizes must be positive")
        if self.cache_entries < 1:
            raise ValueError("cache_entries must be positive")


@dataclass(frozen=True)
class ScoredSpan:
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class ReviewDecision:
    start: int
    end: int
    label: str
    categories: tuple[str, ...] = ()
    raw_output: str = ""

    @property
    def should_mask(self) -> bool:
        # Qwen3Guard-Gen is categorical, not threshold based. A malformed reply
        # is failed closed because this text already crossed the primary guard.
        # Qwen can label a direct injection "Controversial + Jailbreak", so the
        # category is authoritative for this injection-specific second stage.
        return (
            self.label.lower() in {"unsafe", "unparseable"}
            or any(category.lower() == "jailbreak" for category in self.categories)
        )


@dataclass(frozen=True)
class MaskedSpan:
    start: int
    end: int
    primary_score: float
    secondary_label: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class SanitizedText:
    text: str
    masked_spans: tuple[MaskedSpan, ...] = ()


class PrimaryScanner(Protocol):
    def suspicious_spans(self, text: str) -> list[ScoredSpan]: ...


class SecondaryReviewer(Protocol):
    def split_windows(self, text: str, start: int, end: int) -> list[tuple[int, int]]: ...

    def review(self, text: str, spans: Sequence[tuple[int, int]]) -> list[ReviewDecision]: ...


def _softmax_positive(logits: Any, positive_index: int) -> list[float]:
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=-1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return [float(row[positive_index]) for row in probabilities]


def _offset_spans(offset_mapping: Any, attention_mask: Any) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for offsets, mask in zip(offset_mapping, attention_mask):
        real = [
            (int(pair[0]), int(pair[1]))
            for pair, enabled in zip(offsets, mask)
            if int(enabled) and int(pair[1]) > int(pair[0])
        ]
        spans.append((min(a for a, _ in real), max(b for _, b in real)) if real else (0, 0))
    return spans


class WolfDefenderOnnxScanner:
    """Lazy CPU ONNX runner for Wolf Defender Small v2."""

    def __init__(self, config: PromptInjectionGuardConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._session = None
        self._positive_index = 1
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoConfig, AutoTokenizer

            cache_dir = str(self.config.cache_dir)
            model_path = hf_hub_download(
                repo_id=self.config.primary_model,
                filename=self.config.primary_onnx_file,
                cache_dir=cache_dir,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.primary_model,
                cache_dir=cache_dir,
                use_fast=True,
            )
            if not getattr(tokenizer, "is_fast", False):
                raise RuntimeError("Wolf Defender requires a fast tokenizer for source offsets")
            model_config = AutoConfig.from_pretrained(
                self.config.primary_model,
                cache_dir=cache_dir,
            )
            labels = {int(key): str(value).upper() for key, value in model_config.id2label.items()}
            positive = next((key for key, value in labels.items() if value == "INJECTION"), 1)
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._tokenizer = tokenizer
            self._session = session
            self._positive_index = positive

    def suspicious_spans(self, text: str) -> list[ScoredSpan]:
        if not text.strip():
            return []
        self._ensure_loaded()
        import numpy as np

        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self.config.primary_window_tokens,
            stride=self.config.primary_overlap_tokens,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np",
            padding=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        spans = _offset_spans(offsets, encoded["attention_mask"])
        input_names = {entry.name for entry in self._session.get_inputs()}
        scores: list[float] = []
        total = int(encoded["input_ids"].shape[0])
        for first in range(0, total, self.config.primary_batch_size):
            last = min(total, first + self.config.primary_batch_size)
            inputs = {
                name: np.asarray(value[first:last], dtype=np.int64)
                for name, value in encoded.items()
                if name in input_names
            }
            logits = self._session.run(None, inputs)[0]
            scores.extend(_softmax_positive(logits, self._positive_index))
        return [
            ScoredSpan(start, end, score)
            for (start, end), score in zip(spans, scores)
            if end > start and score >= self.config.primary_threshold
        ]


class Qwen3GuardReviewer:
    """Lazy GPU/CPU categorical reviewer for primary-positive 512-token blocks."""

    _LABEL = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
    _CATEGORIES = re.compile(
        r"Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
        r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
        r"Copyright Violation|Jailbreak|None",
        re.IGNORECASE,
    )

    def __init__(self, config: PromptInjectionGuardConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cache_dir = str(self.config.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.secondary_model,
                cache_dir=cache_dir,
                use_fast=True,
            )
            if not getattr(tokenizer, "is_fast", False):
                raise RuntimeError("Qwen3Guard requires a fast tokenizer for source offsets")
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            device = self.config.secondary_device
            resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if resolved_device == "auto":
                resolved_device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.config.secondary_model,
                cache_dir=cache_dir,
                torch_dtype="auto",
            ).to(resolved_device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def split_windows(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        self._ensure_loaded()
        segment = text[start:end]
        encoded = self._tokenizer(
            segment,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.secondary_window_tokens,
            stride=self.config.secondary_overlap_tokens,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        mappings = encoded["offset_mapping"]
        windows: list[tuple[int, int]] = []
        for offsets in mappings:
            real = [(int(a), int(b)) for a, b in offsets if int(b) > int(a)]
            if real:
                windows.append((start + min(a for a, _ in real), start + max(b for _, b in real)))
        return windows

    def review(self, text: str, spans: Sequence[tuple[int, int]]) -> list[ReviewDecision]:
        if not spans:
            return []
        self._ensure_loaded()
        import torch

        decisions: list[ReviewDecision] = []
        for first in range(0, len(spans), self.config.secondary_batch_size):
            current = spans[first : first + self.config.secondary_batch_size]
            rendered = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": text[start:end]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for start, end in current
            ]
            model_inputs = self._tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
            ).to(self._model.device)
            with torch.inference_mode():
                outputs = self._model.generate(
                    **model_inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            generated = outputs[:, model_inputs["input_ids"].shape[1] :]
            replies = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (start, end), reply in zip(current, replies):
                match = self._LABEL.search(reply)
                label = match.group(1).title() if match else "Unparseable"
                categories = tuple(
                    dict.fromkeys(value.title() for value in self._CATEGORIES.findall(reply) if value.lower() != "none")
                )
                decisions.append(ReviewDecision(start, end, label, categories, reply[:500]))
        return decisions


def _merge_scored_spans(spans: Sequence[ScoredSpan]) -> list[ScoredSpan]:
    merged: list[ScoredSpan] = []
    for current in sorted(spans, key=lambda item: (item.start, item.end)):
        if merged and current.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = ScoredSpan(
                previous.start,
                max(previous.end, current.end),
                max(previous.score, current.score),
            )
        else:
            merged.append(current)
    return merged


def _score_for_span(primary: Sequence[ScoredSpan], start: int, end: int) -> float:
    overlapping = [item.score for item in primary if item.start < end and item.end > start]
    return max(overlapping, default=0.0)


def _merge_masked_spans(spans: Sequence[MaskedSpan]) -> list[MaskedSpan]:
    merged: list[MaskedSpan] = []
    for current in sorted(spans, key=lambda item: (item.start, item.end)):
        if merged and current.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = MaskedSpan(
                previous.start,
                max(previous.end, current.end),
                max(previous.primary_score, current.primary_score),
                "Unsafe" if "unsafe" in {previous.secondary_label.lower(), current.secondary_label.lower()} else current.secondary_label,
                tuple(dict.fromkeys((*previous.categories, *current.categories))),
            )
        else:
            merged.append(current)
    return merged


def _marker(span: MaskedSpan) -> str:
    categories = "、".join(span.categories) if span.categories else "未提供"
    return (
        "[疑似恶意外部内容已遮蔽；"
        f"一级检测分数={span.primary_score:.3f}；"
        f"二级复核={span.secondary_label}；类别={categories}]"
    )


class PromptInjectionGuard:
    """Two-stage scanner with exact source-span replacement and an LRU cache."""

    def __init__(
        self,
        config: PromptInjectionGuardConfig,
        *,
        primary: PrimaryScanner | None = None,
        secondary: SecondaryReviewer | None = None,
    ) -> None:
        self.config = config
        self.primary = primary or WolfDefenderOnnxScanner(config)
        self.secondary = secondary or Qwen3GuardReviewer(config)
        self._cache: OrderedDict[str, SanitizedText] = OrderedDict()
        self._cache_lock = threading.Lock()
        # ONNX and generation calls are batched inside one shared model instance.
        self._inference_lock = threading.Lock()

    def _cache_key(self, text: str) -> str:
        policy = (
            f"v1|{self.config.primary_model}|{self.config.primary_threshold}|"
            f"{self.config.primary_window_tokens}|{self.config.primary_overlap_tokens}|"
            f"{self.config.secondary_model}|{self.config.secondary_window_tokens}|"
            f"{self.config.secondary_overlap_tokens}"
        )
        return hashlib.sha256((policy + "\0" + text).encode("utf-8")).hexdigest()

    def sanitize_text(self, text: str) -> SanitizedText:
        if not self.config.enabled or not text.strip():
            return SanitizedText(text)
        key = self._cache_key(text)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        with self._inference_lock:
            primary_hits = self.primary.suspicious_spans(text)
            candidate_spans: list[tuple[int, int]] = []
            for hit in _merge_scored_spans(primary_hits):
                candidate_spans.extend(self.secondary.split_windows(text, hit.start, hit.end))
            # Overlapping primary windows can yield identical secondary blocks.
            candidate_spans = list(dict.fromkeys(candidate_spans))
            reviews = self.secondary.review(text, candidate_spans)
        masked = _merge_masked_spans(
            [
                MaskedSpan(
                    decision.start,
                    decision.end,
                    _score_for_span(primary_hits, decision.start, decision.end),
                    decision.label,
                    decision.categories,
                )
                for decision in reviews
                if decision.should_mask
            ]
        )
        output = text
        for span in reversed(masked):
            output = output[: span.start] + _marker(span) + output[span.end :]
        result = SanitizedText(output, tuple(masked))
        with self._cache_lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.config.cache_entries:
                self._cache.popitem(last=False)
        return result


def _sanitize_json_value(value: Any, guard: PromptInjectionGuard) -> tuple[Any, int]:
    if isinstance(value, str):
        result = guard.sanitize_text(value)
        return result.text, len(result.masked_spans)
    if isinstance(value, list):
        rewritten = []
        count = 0
        for item in value:
            clean, masked = _sanitize_json_value(item, guard)
            rewritten.append(clean)
            count += masked
        return rewritten, count
    if isinstance(value, dict):
        rewritten = {}
        count = 0
        for key, item in value.items():
            if isinstance(item, str) and str(key).lower() in _STRUCTURAL_STRING_KEYS:
                clean, masked = item, 0
            else:
                clean, masked = _sanitize_json_value(item, guard)
            rewritten[key] = clean
            count += masked
        return rewritten, count
    return value, 0


def sanitize_message_content(content: Any, guard: PromptInjectionGuard) -> tuple[Any, int]:
    """Sanitize strings without flattening structured MCP/JSON content."""

    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            result = guard.sanitize_text(content)
            return result.text, len(result.masked_spans)
        if isinstance(decoded, (dict, list)):
            clean, count = _sanitize_json_value(decoded, guard)
            return json.dumps(clean, ensure_ascii=False), count
        result = guard.sanitize_text(content)
        return result.text, len(result.masked_spans)
    return _sanitize_json_value(content, guard)


def _sanitize_tool_message(message: ToolMessage, guard: PromptInjectionGuard) -> ToolMessage:
    content, masked = sanitize_message_content(message.content, guard)
    if not masked:
        return message
    metadata = dict(message.response_metadata)
    metadata["prompt_injection_guard"] = {"masked_blocks": masked, "policy": "wolf-small->qwen3guard-0.6b"}
    return message.model_copy(update={"content": content, "response_metadata": metadata})


def sanitize_tool_result(result: ToolMessage | Command[Any], guard: PromptInjectionGuard):
    if isinstance(result, ToolMessage):
        return _sanitize_tool_message(result, guard)
    if isinstance(result, Command) and isinstance(result.update, dict):
        update = dict(result.update)
        messages = update.get("messages")
        if isinstance(messages, list):
            update["messages"] = [
                _sanitize_tool_message(message, guard) if isinstance(message, ToolMessage) else message
                for message in messages
            ]
            return Command(graph=result.graph, update=update, resume=result.resume, goto=result.goto)
    return result


class PromptInjectionGuardMiddleware(AgentMiddleware):
    """Intercept every tool result before its text is exposed to a Worker model."""

    def __init__(self, guard: PromptInjectionGuard) -> None:
        self.guard = guard

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        return await asyncio.to_thread(sanitize_tool_result, result, self.guard)


__all__ = [
    "MaskedSpan",
    "PromptInjectionGuard",
    "PromptInjectionGuardConfig",
    "PromptInjectionGuardMiddleware",
    "ReviewDecision",
    "SanitizedText",
    "ScoredSpan",
    "sanitize_message_content",
    "sanitize_tool_result",
]
