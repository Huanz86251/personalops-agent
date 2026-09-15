"""Local intent routing and bounded semantic scope resolution."""
from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from typing import Any, Mapping

from pydantic import ValidationError

from observability import set_span_attributes, set_span_output, trace_span
from planning_models import PlanningContextPack, ScopeContract
from prompt_loader import load_prompt, render_prompt
from scope_intent_classifier import ScopeIntentClassifier


DEFAULT_SCOPE_RESOLVER_MAX_ATTEMPTS = 3


class ScopeResolutionFailure(ValueError):
    """A resolver failure that retains the number of paid model attempts."""

    def __init__(self, message: str, *, attempts: int, error_type: str | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.error_type = error_type or type(self).__name__


@lru_cache(maxsize=1)
def _classifier() -> ScopeIntentClassifier:
    return ScopeIntentClassifier()


async def classify_scope_request(user_request: str) -> dict[str, Any]:
    """Run the local router; failure conservatively requests cloud resolution."""
    try:
        result = await asyncio.to_thread(_classifier().predict, user_request)
        return {**result, "router_status": "ok"}
    except Exception as error:
        return {
            "label": "REQUIRES_SCOPE_CONTRACT",
            "label_id": 1,
            "probability_requires_scope": None,
            "threshold": None,
            "router_status": "fallback_requires_scope",
            "error_type": type(error).__name__,
        }


def _max_attempts() -> int:
    name = "SCOPE_RESOLVER_MAX_ATTEMPTS"
    try:
        value = int(os.getenv(name, str(DEFAULT_SCOPE_RESOLVER_MAX_ATTEMPTS)))
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None
    if not 1 <= value <= 3:
        raise ValueError(f"{name} must be between 1 and 3")
    return value


def _raw_content(response: Mapping[str, Any]) -> str:
    raw = response.get("raw")
    content = raw.get("content") if isinstance(raw, Mapping) else getattr(raw, "content", None)
    return content if isinstance(content, str) else ""


def _finish_reason(response: Mapping[str, Any]) -> str:
    raw = response.get("raw")
    if isinstance(raw, Mapping):
        metadata = raw.get("response_metadata") or {}
        direct = raw.get("finish_reason")
    else:
        metadata = getattr(raw, "response_metadata", {}) or {}
        direct = getattr(raw, "finish_reason", None)
    if isinstance(metadata, Mapping):
        direct = metadata.get("finish_reason") or metadata.get("stop_reason") or direct
    return str(direct or "").strip().lower()


def _validation_details(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        lines = []
        for item in error.errors(include_url=False):
            path = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
            lines.append(
                f"{path}: {item.get('type', 'validation_error')} - {item.get('msg', 'invalid value')}"
            )
        return "\n".join(lines)[:2000]
    return f"{type(error).__name__}: {str(error)}"[:2000]


def _is_truncation(response: Mapping[str, Any], error: BaseException | None) -> bool:
    if _finish_reason(response) in {"length", "max_tokens", "max_output_tokens", "incomplete"}:
        return True
    text = str(error or "").lower()
    return any(marker in text for marker in (
        "max token", "maximum token", "token limit", "length limit", "finish_reason='length'",
        'finish_reason="length"', "output was truncated",
    ))


def _is_structured_output_error(error: BaseException) -> bool:
    """Identify model-output parsing failures without retrying transport errors."""
    if isinstance(error, (ValidationError, json.JSONDecodeError)):
        return True
    error_name = type(error).__name__.lower()
    text = str(error).lower()
    return (
        "outputparser" in error_name
        or "parse" in error_name and "response" in text
        or "validation error" in text and "scope" in text
    )


def _repair_message(
    *, failure_kind: str, failure_details: str, previous_output: str,
) -> dict[str, str]:
    return {
        "role": "user",
        "content": render_prompt(
            "planning/scope_resolver_repair",
            failure_kind=failure_kind,
            failure_details=failure_details,
            previous_output=previous_output[:12000] or "（没有可复用的完整输出）",
        ),
    }


async def resolve_scope_contract(
    model,
    user_request: str,
    current_time: str = "unknown",
) -> tuple[ScopeContract, int]:
    """Resolve a contract, retrying only truncated or invalid structured output."""
    messages = [
        {"role": "system", "content": render_prompt(
            "planning/scope_resolver_system",
            schema=json.dumps(
                ScopeContract.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )},
        {"role": "user", "content": render_prompt(
            "planning/scope_resolver_request",
            current_time=current_time,
            user_request=user_request,
        )},
    ]
    structured = model.with_structured_output(
        ScopeContract, method="json_mode", include_raw=True,
    )
    attempts = 0
    conversation = list(messages)
    last_error: BaseException | None = None
    for attempt in range(1, _max_attempts() + 1):
        attempts = attempt
        try:
            response = await structured.ainvoke(conversation)
        except Exception as error:
            truncated = _is_truncation({}, error)
            if truncated or _is_structured_output_error(error):
                response = {
                    "raw": {
                        "content": "",
                        "response_metadata": {
                            "finish_reason": "length" if truncated else "",
                        },
                    },
                    "parsed": None,
                    "parsing_error": error,
                }
                last_error = error
            else:
            # Transport/provider failures remain governed by the provider retry setting.
                raise ScopeResolutionFailure(
                    f"{type(error).__name__}: {str(error)[:1200]}", attempts=attempts,
                    error_type=type(error).__name__,
                ) from error

        try:
            if isinstance(response, ScopeContract):
                return response, attempts
            if not isinstance(response, Mapping):
                raise TypeError("Scope Resolver did not return a structured mapping")
            parsing_error = response.get("parsing_error")
            if parsing_error is not None:
                raw_content = _raw_content(response)
                if raw_content.strip():
                    contract = ScopeContract.model_validate_json(raw_content)
                    return contract, attempts
                raise parsing_error
            return ScopeContract.model_validate(response.get("parsed")), attempts
        except Exception as error:
            last_error = error
            if attempt >= _max_attempts():
                break
            truncated = _is_truncation(response if isinstance(response, Mapping) else {}, error)
            if truncated:
                failure_kind = "输出被截断，尚未形成可用的 ScopeContract"
                details = (
                    "上一轮推理或输出过长。请显著缩短思考过程，直接输出完整 JSON；"
                    "不要解释，不要重复任务背景。"
                )
                previous_output = ""
            else:
                failure_kind = "ScopeContract 格式或字段校验失败"
                details = _validation_details(error)
                previous_output = _raw_content(response) if isinstance(response, Mapping) else ""
            conversation.append(_repair_message(
                failure_kind=failure_kind,
                failure_details=details,
                previous_output=previous_output,
            ))

    raise ScopeResolutionFailure(
        _validation_details(last_error or ValueError("unknown scope resolution failure")),
        attempts=attempts,
    ) from last_error


async def prepare_scope_context(
    context: PlanningContextPack,
    model,
) -> tuple[PlanningContextPack, int]:
    """Freeze one local route and, when selected, one cloud ScopeContract."""
    if context.scope_router is not None:
        return context, 0

    with trace_span(
        "Scope Router / local intent classification",
        kind="chain",
        input_value={"request_chars": len(context.user_request)},
    ) as router_span:
        route = await classify_scope_request(context.user_request)
        set_span_attributes(router_span, **{
            "scope.router_label": route["label"],
            "scope.router_status": route["router_status"],
        })
        set_span_output(router_span, route)

    contract = None
    model_calls = 0
    if route["label"] == "REQUIRES_SCOPE_CONTRACT":
        try:
            with trace_span(
                "Scope Resolver / Qwen structured contract",
                kind="chain",
                input_value={
                    "schema": "ScopeContract",
                    "request_chars": len(context.user_request),
                },
            ) as resolver_span:
                contract, model_calls = await resolve_scope_contract(
                    model,
                    context.user_request,
                    context.current_time,
                )
                set_span_attributes(
                    resolver_span, **{
                        "scope.resolution_status": "ok",
                        "scope.resolution_attempts": model_calls,
                    }
                )
                set_span_output(resolver_span, contract)
            route = {**route, "resolver_status": "ok"}
        except Exception as error:
            model_calls = getattr(error, "attempts", model_calls or 0)
            route = {
                **route,
                "resolver_status": "failed",
                "resolver_error_type": getattr(error, "error_type", type(error).__name__),
                "resolver_error": str(error)[:1200],
                "resolver_attempts": model_calls,
            }

    return context.model_copy(update={
        "scope_router": route,
        "scope_contract": contract,
    }), model_calls
