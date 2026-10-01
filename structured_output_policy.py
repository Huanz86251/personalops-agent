"""Provider-aware structured-output policy.

The Harness still validates every parsed object.  This module only chooses the
strongest generation-time contract that the configured provider/model has
demonstrated it can honor.
"""

from __future__ import annotations

import json
from typing import Any

from schema_utils import compact_schema


_STRICT_JSON_SCHEMA_MODEL_PREFIXES = (
    "qwen3.7-flash",
    "qwen3.7-plus",
    "qwen3.7-max",
    "qwen3.8-flash",
    "qwen3.8-max",
)
_PROBED_STRICT_JSON_SCHEMA_MODELS = {
    # Verified against the current Bailian OpenAI-compatible endpoint on
    # 2026-09-23 with response_format=json_schema and strict=true.  Keep this
    # exact so a future FlashX or other sibling is not enabled by inference.
    "zhipu/glm-5.3-flash",
}


def model_identifier(model: Any) -> str:
    """Return a stable lower-case model identifier without inspecting secrets."""

    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def supports_strict_json_schema(model: Any) -> bool:
    """Whether this exact model family has provider-side strict-schema support."""

    identifier = model_identifier(model)
    return identifier in _PROBED_STRICT_JSON_SCHEMA_MODELS or any(
        identifier.startswith(prefix)
        for prefix in _STRICT_JSON_SCHEMA_MODEL_PREFIXES
    )


def with_provider_structured_output(
    model: Any,
    schema: Any,
    *,
    include_raw: bool,
    fallback_method: str | None = "json_mode",
    tools: list[Any] | None = None,
):
    """Bind a schema while preserving the existing parser and fallback layers.

    Qwen3.7/3.8 and the probed Bailian GLM-5.3-Flash path receive native strict
    JSON Schema.  Other models keep the call site's previous method so this
    policy does not silently change unverified providers.
    """

    if supports_strict_json_schema(model):
        options: dict[str, Any] = {
            "method": "json_schema",
            "strict": True,
            "include_raw": include_raw,
        }
        if tools:
            options["tools"] = tools
        return model.with_structured_output(schema, **options)

    # Preserve the exact legacy call shape for unverified providers.  In
    # particular, callers that previously relied on the default
    # include_raw=False should not suddenly receive a new keyword that a test
    # double or provider adapter may not accept.
    options: dict[str, Any] = {}
    if include_raw:
        options["include_raw"] = True
    if fallback_method is not None:
        options["method"] = fallback_method
    return model.with_structured_output(schema, **options)


def strict_tool_schema_options(model: Any) -> dict[str, bool]:
    """Enable strict function-argument schemas for verified model families."""

    return {"strict": True} if supports_strict_json_schema(model) else {}


def textual_schema_fallback(
    model: Any,
    schema: Any,
    *,
    heading: str = "Schema:",
) -> str:
    """Return a text copy of a schema only when the provider cannot enforce it.

    Verified Qwen/GLM routes already receive the complete contract through
    ``response_format=json_schema`` or a strict function definition. Repeating
    that same object in the prompt is expensive and, for large planning
    schemas, can expose the model to several competing copies of one protocol.
    Unknown providers retain the legacy prompt copy as a compatibility
    fallback.
    """

    if supports_strict_json_schema(model):
        return ""
    raw_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
    return heading + json.dumps(
        compact_schema(raw_schema),
        ensure_ascii=False,
        separators=(",", ":"),
    )
