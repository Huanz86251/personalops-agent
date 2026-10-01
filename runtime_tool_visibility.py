"""Async-local summary of the tools mounted on the current model request."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Sequence


VISIBLE_MODEL_TOOLS: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "visible_model_tools",
    default=(),
)


def _parameter_summary(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if schema is not None and hasattr(schema, "model_json_schema"):
        try:
            schema = schema.model_json_schema()
        except Exception:
            schema = None
    if not isinstance(schema, dict):
        schema = getattr(tool, "args", None)
    if not isinstance(schema, dict):
        return {"properties": [], "required": []}
    properties = schema.get("properties") if "properties" in schema else schema
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    return {
        "properties": [str(name) for name in properties][:32],
        "required": [str(name) for name in required][:32],
    }


def summarize_visible_tools(tools: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Return a bounded, credential-free view of the exact mounted tool list."""

    rows: list[dict[str, Any]] = []
    for tool in tools[:64]:
        description = str(getattr(tool, "description", "") or "").strip()
        if len(description) > 360:
            description = description[:357] + "..."
        rows.append({
            "name": str(getattr(tool, "name", "") or "").strip(),
            "description": description,
            **_parameter_summary(tool),
        })
    return tuple(rows)


@contextmanager
def visible_model_tools_scope(tools: Sequence[Any]) -> Iterator[tuple[dict[str, Any], ...]]:
    summary = summarize_visible_tools(tools)
    token = VISIBLE_MODEL_TOOLS.set(summary)
    try:
        yield summary
    finally:
        VISIBLE_MODEL_TOOLS.reset(token)
