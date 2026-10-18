"""Shared strict-schema parsing, harmless repair and one bounded retry."""
import json

from pydantic import BaseModel

from structured_output_policy import with_provider_structured_output


class CompactionFailure(ValueError):
    def __init__(self, reason, attempts):
        super().__init__(reason)
        self.attempts = attempts


def validate_summary_result(result, schema, *, core_fields, max_chars):
    raw = result.get("raw")
    metadata = getattr(raw, "response_metadata", {}) or {}
    if metadata.get("finish_reason") in {"length", "max_tokens"}:
        raise ValueError("truncated_output")
    value = result.get("parsed")
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if value is None:
        content = getattr(raw, "content", None)
        if not isinstance(content, str):
            raise ValueError("missing_json_object")
        try:
            value = json.loads(content)
        except ValueError:
            raise ValueError("invalid_json") from None
    if not isinstance(value, dict):
        raise ValueError("expected_json_object")
    normalized = {}
    for name in schema.model_fields:
        item = value.get(name)
        # Empty/missing slots and scalar text are deterministic container repair.
        if item is None or item == "":
            item = []
        elif isinstance(item, str):
            item = [item]
        if isinstance(item, list):
            item = [entry.strip() for entry in item if isinstance(entry, str) and entry.strip()]
            # Preserve invalid nonempty core items as an error, not an empty success.
            original = value.get(name)
            if name in core_fields and isinstance(original, list) and any(
                entry is not None and not isinstance(entry, str) for entry in original
            ):
                raise ValueError("invalid_core_field:" + name)
        elif name not in core_fields:
            item = []  # Optional malformed metadata does not trigger a model repair.
        else:
            raise ValueError("invalid_core_field:" + name)
        normalized[name] = item
    summary = schema.model_validate(normalized)
    if not any(normalized[name] for name in core_fields):
        raise ValueError("empty_substantive_summary")
    if len(summary.model_dump_json()) > max_chars:
        raise ValueError("summary_size_exceeded")
    return summary


def _repair_messages(messages, reason):
    # Only controlled validation codes; do not echo private model output or traces.
    return [*messages, {"role": "user", "content": (
        "上次压缩输出未通过检查：" + reason
        + "。请基于同一份输入重新输出符合接口Schema的紧凑JSON。"
        "实质结果、要求或未完成事项不能全部为空；无材料的可选字段使用空数组。"
    )}]


def _reason(error):
    text = str(error)
    allowed = {"truncated_output", "missing_json_object", "invalid_json", "expected_json_object",
               "empty_substantive_summary", "summary_size_exceeded", "summary_did_not_shrink"}
    if text in allowed or (text.startswith(("invalid_core_field:", "summary_budget_exceeded:actual=")) and len(text) < 100):
        return text
    return "generation_or_validation_failed"


def run_compaction(model, schema, messages, *, core_fields, max_chars, check=None):
    request = messages
    for attempt in range(1, 3):
        try:
            # A schema dict postpones Pydantic parsing until harmless Harness
            # normalization, while keeping the exact cloud-side strict contract.
            result = with_provider_structured_output(model, schema.model_json_schema(), include_raw=True).invoke(request)
            summary = validate_summary_result(result, schema, core_fields=core_fields, max_chars=max_chars)
            if check is not None:
                check(summary)
            return summary, attempt
        except Exception as error:
            reason = _reason(error)
            if attempt == 2:
                raise CompactionFailure(reason, attempt) from None
            request = _repair_messages(messages, reason)


async def arun_compaction(model, schema, messages, *, core_fields, max_chars, check=None):
    request = messages
    for attempt in range(1, 3):
        try:
            result = await with_provider_structured_output(model, schema.model_json_schema(), include_raw=True).ainvoke(request)
            summary = validate_summary_result(result, schema, core_fields=core_fields, max_chars=max_chars)
            if check is not None:
                check(summary)
            return summary, attempt
        except Exception as error:
            reason = _reason(error)
            if attempt == 2:
                raise CompactionFailure(reason, attempt) from None
            request = _repair_messages(messages, reason)
