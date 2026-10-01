"""Harness arbitration for terminal report tools.

Business/read calls and a terminal report emitted in the same assistant turn
cannot describe the same state: the report was authored before those calls
returned. ToolNode may still execute useful sibling calls concurrently; only
the premature terminal call is deterministically deferred.
"""

from __future__ import annotations

from typing import Any

from langchain.messages import ToolMessage
from langgraph.types import Command


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _tool_name(call: Any) -> str:
    name = _value(call, "name", "")
    if not name:
        name = _value(_value(call, "function", {}), "name", "")
    return str(name or "").strip()


def defer_terminal_call_if_mixed(runtime: Any, terminal_name: str) -> Command | None:
    """Reject only the report call when its round also contains other tools."""

    state = _value(runtime, "state", {}) or {}
    messages = list(state.get("messages", []) or [])
    if not messages:
        return None
    calls = list(_value(messages[-1], "tool_calls", []) or [])
    sibling_names = [
        name
        for call in calls
        if (name := _tool_name(call)) and name != terminal_name
    ]
    if not sibling_names:
        return None
    unique_siblings = list(dict.fromkeys(sibling_names))
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        f"{terminal_name} was deferred because the same model round also "
                        f"requested: {', '.join(unique_siblings)}. Those calls may execute "
                        "now. Inspect their returned ToolMessages, then submit the report "
                        "alone in the next round. Do not repeat calls that already succeeded."
                    ),
                    tool_call_id=(
                        _value(runtime, "tool_call_id", None) or terminal_name
                    ),
                    name=terminal_name,
                    status="error",
                )
            ]
        }
    )


__all__ = ["defer_terminal_call_if_mixed"]
