"""Small in-memory BM25 index for active typed memories.

SQLite remains the source of truth. This index is rebuilt at startup and kept
in sync with the existing NetworkX side index.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from threading import RLock
from typing import Any
import unicodedata


_TOKEN_PATTERN = re.compile(
    r"\w+(?:[.+#-]\w+)*",
    re.UNICODE,
)

_STRUCTURED_TEXT_FIELDS = (
    "record_type",
    "field",
    "value",
    "preference",
    "topic",
    "scope",
    "scope_name",
    "person_name",
    "relation",
    "other_relation",
    "state",
    "project_name",
    "fact",
    "event",
    "from_person",
    "to_person",
    "action",
    "object",
    "status",
)


@dataclass(frozen=True)
class BM25MemoryHit:
    memory_id: str
    score: float


def tokenize_memory_text(text: str) -> list[str]:
    """Tokenize mixed Chinese/Latin text without a resident segmenter."""

    normalized = unicodedata.normalize("NFC", str(text or "")).casefold()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        token = match.group(0)
        if any("\u3400" <= char <= "\u9fff" for char in token):
            if len(token) == 1:
                tokens.append(token)
            else:
                tokens.extend(
                    token[index:index + 2]
                    for index in range(len(token) - 1)
                )
                if len(token) <= 12:
                    tokens.append(token)
        else:
            tokens.append(token)
    return tokens


def memory_lexical_document(value: dict[str, Any]) -> str:
    """Build a compact lexical document from summary and controlled fields."""

    content = str(value.get("content", "") or "").strip()
    parts = [content, content] if content else []

    structured_data = value.get("structured_data", {})
    if isinstance(structured_data, dict):
        for field_name in _STRUCTURED_TEXT_FIELDS:
            field_value = structured_data.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                parts.append(field_value.strip())

    triples = value.get("triples", [])
    if isinstance(triples, list):
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            for field_name in ("subject", "relation", "object"):
                field_value = triple.get(field_name)
                if isinstance(field_value, str) and field_value.strip():
                    parts.append(field_value.strip())

    return "\n".join(parts)


class MemoryBM25Index:
    """Thread-safe BM25 index over the active personal-memory corpus."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[str, Counter[str]] = {}
        self._document_lengths: dict[str, int] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._total_document_length = 0
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._document_lengths.clear()
            self._document_frequencies.clear()
            self._total_document_length = 0

    def add_memory(self, memory_id: str, value: dict[str, Any]) -> bool:
        if not memory_id or value.get("status") != "active":
            return False

        tokens = tokenize_memory_text(memory_lexical_document(value))
        with self._lock:
            self.remove_memory(memory_id)
            if not tokens:
                return False
            frequencies = Counter(tokens)
            self._documents[memory_id] = frequencies
            self._document_lengths[memory_id] = len(tokens)
            self._total_document_length += len(tokens)
            self._document_frequencies.update(frequencies.keys())
            return True

    def remove_memory(self, memory_id: str) -> bool:
        with self._lock:
            frequencies = self._documents.pop(memory_id, None)
            length = self._document_lengths.pop(memory_id, 0)
            if frequencies is None:
                return False
            self._total_document_length -= length
            for token in frequencies:
                remaining = self._document_frequencies[token] - 1
                if remaining > 0:
                    self._document_frequencies[token] = remaining
                else:
                    del self._document_frequencies[token]
            return True

    def search(self, query: str, *, limit: int = 8) -> list[BM25MemoryHit]:
        query_tokens = set(tokenize_memory_text(query))
        if not query_tokens or limit < 1:
            return []

        with self._lock:
            document_count = len(self._documents)
            if document_count == 0:
                return []
            average_length = self._total_document_length / document_count
            scores: list[BM25MemoryHit] = []
            for memory_id, frequencies in self._documents.items():
                document_length = self._document_lengths[memory_id]
                score = 0.0
                for token in query_tokens:
                    term_frequency = frequencies.get(token, 0)
                    if term_frequency == 0:
                        continue
                    document_frequency = self._document_frequencies[token]
                    inverse_document_frequency = math.log(
                        1.0
                        + (document_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                    )
                    denominator = term_frequency + self.k1 * (
                        1.0
                        - self.b
                        + self.b * document_length / average_length
                    )
                    score += inverse_document_frequency * (
                        term_frequency * (self.k1 + 1.0) / denominator
                    )
                if score > 0.0:
                    scores.append(BM25MemoryHit(memory_id=memory_id, score=score))

            scores.sort(key=lambda hit: (-hit.score, hit.memory_id))
            return scores[:limit]

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "documents": len(self._documents),
                "terms": len(self._document_frequencies),
            }
