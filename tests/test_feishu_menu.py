"""Offline checks for the optional in-message Feishu shortcut card."""

from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from lark_channel import new_card


def card_namespace():
    source = Path(__file__).resolve().parents[1] / "main.py"
    names = {"_control_buttons", "_send_control_panel", "_send_card"}
    body = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert len(body) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body],
        type_ignores=[],
    )
    namespace = {
        "new_card": new_card,
        "feishu_channel": SimpleNamespace(send=AsyncMock()),
        "_send_text": AsyncMock(),
        "logger": logging.getLogger("feishu-menu-test"),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)  # noqa: S102
    return namespace


class FeishuMenuTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_card_has_no_empty_button_row(self):
        namespace = card_namespace()
        namespace["feishu_channel"].send.return_value = SimpleNamespace(success=True)
        await namespace["_send_control_panel"]("chat")
        payload = namespace["feishu_channel"].send.call_args.args[1]
        card = payload["card"]
        rows = [element for element in card.data["body"]["elements"] if element["tag"] == "column_set"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["columns"] for row in rows))
        self.assertEqual(sum(len(row["columns"]) for row in rows), 6)
        namespace["_send_text"].assert_not_awaited()

    async def test_rejected_card_logs_error_and_falls_back(self):
        namespace = card_namespace()
        namespace["feishu_channel"].send.return_value = SimpleNamespace(
            success=False,
            error=SimpleNamespace(code="format_error", raw_code=230001, hint="invalid card"),
        )
        with self.assertLogs("feishu-menu-test", level="WARNING") as logs:
            await namespace["_send_control_panel"]("chat")
        self.assertIn("230001", logs.output[0])
        namespace["_send_text"].assert_awaited_once()
