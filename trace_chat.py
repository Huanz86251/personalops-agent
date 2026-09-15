"""Chat presentation of captured messages; never a second model invocation."""
import json


def message_record(message):
    data = message.model_dump(mode="json")
    data["role"] = {"human": "user", "ai": "assistant"}.get(message.type, message.type)
    return data


def emit_code_view(tracer, parent, inputs):
    """Decode the JSON envelope once; never interpret Python string escapes."""
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except (ValueError, TypeError):
            return
    if not isinstance(inputs, dict) or not isinstance(inputs.get("code"), str):
        return
    from opentelemetry.trace import set_span_in_context
    with tracer.start_as_current_span("Code / 代码阅读（仅展示）",
            context=set_span_in_context(parent), openinference_span_kind="chain") as view:
        view.set_attribute("input.value", inputs["code"])
        view.set_attribute("input.mime_type", "text/plain")
        view.set_attribute("audit.display_only", True)
        view.set_attribute("usage.status", "display_only")


def display_json(value, depth=0):
    """Decode JSON containers for display only; never evaluate or repair text."""
    if depth >= 12:
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError, RecursionError):
            prefix, sep, body = value.partition("\n")
            if not sep:
                return value
            try:
                decoded = json.loads(body)
            except (ValueError, TypeError, RecursionError):
                return value
            if isinstance(decoded, (dict, list)):
                return {"原文前缀或路径": prefix, "正文": display_json(decoded, depth+1)}
            return value
        if isinstance(decoded, (dict, list)):
            return display_json(decoded, depth+1)
        return value
    if isinstance(value, dict):
        return {k: display_json(v, depth+1) for k,v in value.items()}
    if isinstance(value, list):
        return [display_json(v, depth+1) for v in value]
    return value


def _json_block(value):
    text = json.dumps(display_json(value), ensure_ascii=False, indent=2)
    # A payload cannot close its display fence.
    fence = "`" * max(3, max((len(part) for part in __import__("re").findall(r"`+", text)), default=0) + 1)
    return fence + "json\n" + text + "\n" + fence


def readable_content(content):
    """Pure rendering only: originals remain in input.value/output.value."""
    value = display_json(content)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return content
    if not isinstance(value, (dict, list)):
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if isinstance(value, dict) and ("协议" in value or "使用协议" in value):
        blocks = []
        for key, item in value.items():
            if key == "要求" and isinstance(item, str):
                explanation, marker, schema = item.partition("Schema:")
                blocks.append("### 填写规则\n\n" + explanation)
                if marker:
                    try:
                        blocks.append("### Schema · 字段定义\n\n" + _json_block(json.loads(schema)))
                    except ValueError:
                        blocks.append("### Schema · 原文（无法解析）\n\n" + _json_block(schema))
            else:
                blocks.append("### " + key + "\n\n" + _json_block(item))
        return "\n\n".join(blocks)
    return _json_block(value)


def chat_attributes(inputs, output):
    attrs = {"audit.display_only": True, "usage.status": "display_only",
             "audit.capture_boundary": "LangChain callback messages, not raw provider HTTP; unexposed provider fields are unavailable"}
    incoming = [m for batch in inputs.get("messages", []) for m in batch]
    outgoing = output.get("responses", [])
    for direction, messages in (("input", incoming), ("output", outgoing)):
        for i, message in enumerate(messages):
            prefix = f"llm.{direction}_messages.{i}.message."
            role = message.get("role", message.get("type", "assistant" if direction == "output" else "user"))
            attrs[prefix + "role"] = {"human": "user", "ai": "assistant"}.get(role, role)
            content = message.get("content", "")
            attrs[prefix + "content"] = readable_content(content)
            reasoning = message.get("additional_kwargs", {}).get("reasoning_content")
            if direction == "output" and isinstance(reasoning, str) and reasoning:
                attrs[prefix + "content"] = ("### 服务商返回的 reasoning\n\n" + reasoning
                    + "\n\n### 模型输出\n\n" + attrs[prefix + "content"])
            for field in ("name", "tool_call_id"):
                if message.get(field): attrs[prefix + field] = message[field]
            for j, call in enumerate(message.get("tool_calls", [])):
                base = prefix + f"tool_calls.{j}.tool_call."
                function = call.get("function", {})
                attrs[base + "id"] = call.get("id") or ""
                attrs[base + "function.name"] = call.get("name", function.get("name", ""))
                args = call.get("args", function.get("arguments", {}))
                attrs[base + "function.arguments"] = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return attrs


def emit_chat_view(tracer, parent, inputs, output):
    from opentelemetry.trace import set_span_in_context
    from observability import set_span_input, set_span_output, set_span_attributes
    with tracer.start_as_current_span("Messages / 完整对话", context=set_span_in_context(parent),
                                      openinference_span_kind="chain") as view:
        set_span_input(view, inputs)
        set_span_output(view, output)
        set_span_attributes(view, **chat_attributes(inputs, output))
