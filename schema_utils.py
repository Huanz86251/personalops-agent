"""Lossless schema guidance with optional display titles removed."""
from copy import deepcopy

from pydantic import ValidationError


def compact_schema(schema):
    """Remove schema titles, preserving guidance, examples and literal data."""
    if isinstance(schema, list):
        return [compact_schema(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key in {"examples", "example", "default", "const", "enum"}:
            # These contain instance data, not schemas; a literal `title`
            # inside an example must reach the model unchanged.
            result[key] = deepcopy(value)
        elif key in {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}:
            result[key] = {name: compact_schema(child) for name, child in value.items()}
        else:
            result[key] = compact_schema(value)
    return result


def minimal_schema_example(schema, *, max_depth=8):
    """Build a compact structural example from JSON Schema without copying failed values."""
    root = schema
    definitions = root.get("$defs", root.get("definitions", {})) if isinstance(root, dict) else {}

    def resolve(node, depth):
        if depth > max_depth or not isinstance(node, dict):
            return None
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return resolve(definitions.get(ref.rsplit("/", 1)[-1], {}), depth + 1)
        if "const" in node:
            return node["const"]
        if node.get("enum"):
            return node["enum"][0]
        for option in node.get("anyOf", node.get("oneOf", [])):
            if isinstance(option, dict) and option.get("type") != "null":
                return resolve(option, depth + 1)
        kind = node.get("type")
        if kind == "object" or "properties" in node:
            properties = node.get("properties", {})
            return {name: resolve(properties.get(name, {}), depth + 1)
                    for name in node.get("required", [])}
        if kind == "array":
            count = max(int(node.get("minItems", 0) or 0), 0)
            return [resolve(node.get("items", {}), depth + 1) for _ in range(count)]
        if "default" in node:
            return node["default"]
        if kind == "boolean":
            return False
        if kind in {"integer", "number"}:
            return node.get("minimum", 0)
        if kind == "null":
            return None
        return "<string>"

    return resolve(root, 0)


def schema_repair_feedback(*, schema_name, schema, error_text, instruction=None, max_chars=2600):
    """Return actionable, value-safe feedback for the same role to repair its form."""
    import json
    required = list(schema.get("required", [])) if isinstance(schema, dict) else []
    example = json.dumps(minimal_schema_example(schema), ensure_ascii=False, separators=(",", ":"))
    message = (
        "SCHEMA_VALIDATION_FAILED\n"
        f"目标Schema：{schema_name}\n"
        f"错误字段：{str(error_text).strip() or '结构化输出未通过校验。'}\n"
        f"顶层必填字段：{', '.join(required) if required else '以当前Schema为准'}\n"
        f"最小结构示例：{example}\n"
        + (instruction or "保留已经得到的业务结果，只修正这张表并重新提交同一个工具；不要重新调用业务API。")
    )
    return message if len(message) <= max_chars else message[:max_chars - 1] + "…"



def is_schema_repairable_error(error):
    """Distinguish malformed structured output from transport/provider failures."""
    current = error
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ValidationError):
            return True
        name = type(current).__name__.lower()
        if any(marker in name for marker in (
            "validation", "parsing", "parser", "jsondecode", "lengthfinishreason"
        )):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    text = f"{type(error).__name__}: {error}".lower()
    transport_markers = (
        "timeout", "timed out", "connection", "rate limit", "ratelimit",
        "service unavailable", "bad gateway", "gateway timeout", "http 502",
        "http 503", "http 504",
    )
    if any(marker in text for marker in transport_markers):
        return False
    return any(marker in text for marker in (
        "json", "schema", "parse", "structured output", "validation"
    ))
