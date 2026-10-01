"""Provider-free checks of the final-answer rich-text delivery boundary."""

from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from lark_channel.channel.outbound import markdown_to_post_ast


def _delivery_namespace():
    source = Path(__file__).resolve().parents[1] / "main.py"
    names = {"_send_text", "_send_final_answer", "_deliver_feishu_event_result"}
    body = [
        node for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name in names
    ]
    assert len(body) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body],
        type_ignores=[],
    )
    namespace = {
        "feishu_channel": SimpleNamespace(send=AsyncMock()),
        "logger": logging.getLogger("feishu-final-reply-test"),
        "FEISHU_EVENT_CONTEXTS": {},
        "EventAction": SimpleNamespace(CANCEL="CANCEL", REPLACE="REPLACE"),
        "_compact_terminal_text": lambda value: value[:40],
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)  # noqa: S102
    return namespace


class FeishuFinalReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_task_uses_rich_text_and_keeps_markdown(self):
        namespace = _delivery_namespace()
        namespace["feishu_channel"].send.return_value = SimpleNamespace(success=True)
        event = SimpleNamespace(event_id="event-1", action="QUEUE", target_event_id=None)
        namespace["FEISHU_EVENT_CONTEXTS"]["event-1"] = ("chat-1", None)

        await namespace["_deliver_feishu_event_result"](event, "  **重点**：第一条\n\n1. 建议  ")

        namespace["feishu_channel"].send.assert_awaited_once_with(
            "chat-1", {"markdown": "**重点**：第一条\n\n1. 建议"},
        )
        self.assertNotIn("event-1", namespace["FEISHU_EVENT_CONTEXTS"])

    async def test_rejected_rich_text_falls_back_to_plain_text(self):
        namespace = _delivery_namespace()
        namespace["feishu_channel"].send.side_effect = [
            SimpleNamespace(
                success=False,
                error=SimpleNamespace(code="invalid", raw_code=230001, hint="bad post"),
            ),
            SimpleNamespace(success=True),
        ]
        with self.assertLogs("feishu-final-reply-test", level="WARNING"):
            await namespace["_send_final_answer"]("chat-1", "**重点**")
        self.assertEqual(
            [call.args[1] for call in namespace["feishu_channel"].send.call_args_list],
            [{"markdown": "**重点**"}, {"text": "**重点**"}],
        )

    async def test_rich_text_exception_falls_back_to_plain_text(self):
        namespace = _delivery_namespace()
        namespace["feishu_channel"].send.side_effect = [RuntimeError("post failure"), SimpleNamespace(success=True)]
        with self.assertLogs("feishu-final-reply-test", level="ERROR"):
            await namespace["_send_final_answer"]("chat-1", "**重点**")
        self.assertEqual(namespace["feishu_channel"].send.call_args_list[-1].args[1], {"text": "**重点**"})

    def test_structured_feishu_post_encodes_bold_without_literal_markers(self):
        post = markdown_to_post_ast("**改进建议**\n\n1. **重点**：内容", tag_md_mode="structured")
        rows = post["zh_cn"]["content"]
        bold = [run for row in rows for run in row if "bold" in run.get("style", [])]
        self.assertEqual([run["text"] for run in bold], ["改进建议", "重点"])
