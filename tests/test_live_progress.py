"""Offline checks for user-visible runtime progress without model narration."""

from __future__ import annotations

import ast
import asyncio
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent import ask_worker
from progress_events import (
    LiveToolProgressHandler, OperationInactivityTimeoutError, ProgressEvent, render_progress_event,
    run_with_progress_keepalive, tool_progress_label,
)
import workers  # Initialize the worker package before importing its Skill middleware.
from skill_runtime.middleware import RoleSkillsMiddleware


@tool
def sample_web_search(query: str) -> str:
    """Return a harmless fixture result."""
    return "fixture search result"


class OneToolModel(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "one-tool-progress-fixture"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        message = (
            AIMessage(content="", tool_calls=[{
                "name": "sample_web_search",
                "args": {"query": "private query must not appear in progress"},
                "id": "fixture-tool-call",
                "type": "tool_call",
            }])
            if self.calls == 1 else AIMessage(content="done")
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class LiveProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_tool_start_reaches_worker_progress_callback(self):
        events: list[ProgressEvent] = []

        async def capture(event: ProgressEvent) -> None:
            events.append(event)

        worker = create_agent(OneToolModel(), tools=[sample_web_search])
        async def run_worker(observed_progress):
            return await ask_worker(
                worker, "Find a result", thread_id="live-progress-fixture",
                state_update={"step_id": "2", "worker_id": "worker-a"},
                runtime_configurable={"progress_callback": observed_progress},
            )
        with patch("knowledge_rag.runtime.automatic_rag", new_callable=AsyncMock):
            await asyncio.wait_for(
                run_with_progress_keepalive(
                    run_worker, capture, interval_seconds=0.01
                ),
                timeout=5.0,
            )
        tool_events = [event for event in events if event.stage == "TOOL_STARTED"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].step_id, 2)
        self.assertNotIn("private query", tool_events[0].message)
        self.assertIn("搜索", tool_events[0].message)

    async def test_repeated_tool_calls_are_throttled_and_do_not_show_arguments(self):
        events: list[ProgressEvent] = []

        async def capture(event: ProgressEvent) -> None:
            events.append(event)

        handler = LiveToolProgressHandler(capture, step_id=1, worker_id="w1")
        await handler.on_tool_start({"name": "web_search"}, "secret=123", run_id="a")
        await handler.on_tool_start({"name": "web_search"}, "secret=456", run_id="b")
        self.assertEqual(len(events), 1)
        self.assertNotIn("secret", events[0].message)
        await handler.on_tool_start(
            {"name": "publish_worker_progress"}, "secret=789", run_id="c"
        )
        self.assertEqual(len(events), 1)

    async def test_model_start_is_reported_without_messages(self):
        events: list[ProgressEvent] = []

        async def capture(event: ProgressEvent) -> None:
            events.append(event)

        handler = LiveToolProgressHandler(capture, step_id=1, worker_id="w1")
        await handler.on_chat_model_start({}, [[]], run_id="model-1")
        self.assertEqual(events[0].stage, "MODEL_STARTED")
        self.assertNotIn("model-1", events[0].message)

    async def test_keepalive_reports_silence_then_stops_at_completion(self):
        events: list[ProgressEvent] = []

        async def capture(event: ProgressEvent) -> None:
            events.append(event)

        async def operation(progress_callback):
            await progress_callback(
                ProgressEvent(stage="STEP_STARTED", message="正在执行", step_id=1)
            )
            await asyncio.sleep(0.045)
            return "finished"

        result = await run_with_progress_keepalive(
            operation, capture, interval_seconds=0.01
        )
        self.assertEqual(result, "finished")
        self.assertGreaterEqual(
            len([event for event in events if event.stage == "ACTIVITY_WAITING"]), 1
        )
        count_at_completion = len(events)
        await asyncio.sleep(0.025)
        self.assertEqual(len(events), count_at_completion)

    async def test_keepalive_stops_after_operation_failure(self):
        events: list[ProgressEvent] = []

        async def capture(event: ProgressEvent) -> None:
            events.append(event)

        async def operation(progress_callback):
            await asyncio.sleep(0.025)
            raise RuntimeError("fixture failure")

        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            await run_with_progress_keepalive(
                operation, capture, interval_seconds=0.01
            )
        count_at_failure = len(events)
        await asyncio.sleep(0.025)
        self.assertEqual(len(events), count_at_failure)

    async def test_keepalive_is_silent_while_waiting_for_user(self):
        events: list[ProgressEvent] = []
        waiting = True

        async def capture(event: ProgressEvent) -> None:
            events.append(event)

        async def operation(progress_callback):
            nonlocal waiting
            await asyncio.sleep(0.035)
            waiting = False
            return "resumed"

        result = await run_with_progress_keepalive(
            operation,
            capture,
            interval_seconds=0.01,
            waiting_for_user=lambda: waiting,
        )
        self.assertEqual(result, "resumed")
        self.assertFalse(
            any(event.stage == "ACTIVITY_WAITING" for event in events)
        )

    async def test_inactivity_watchdog_cancels_only_after_real_silence(self):
        cancelled = asyncio.Event()

        async def operation(_progress_callback):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with self.assertRaises(OperationInactivityTimeoutError):
            await run_with_progress_keepalive(
                operation,
                AsyncMock(),
                interval_seconds=0.01,
                inactivity_timeout_seconds=0.03,
                code_inactivity_timeout_seconds=0.08,
            )
        self.assertTrue(cancelled.is_set())

    async def test_real_progress_keeps_a_long_operation_alive(self):
        async def operation(progress_callback):
            for index in range(5):
                await progress_callback(ProgressEvent(
                    stage="MODEL_STARTED",
                    message="继续执行",
                    status=f"round-{index}",
                ))
                await asyncio.sleep(0.015)
            return "finished"

        result = await run_with_progress_keepalive(
            operation,
            AsyncMock(),
            interval_seconds=0.01,
            inactivity_timeout_seconds=0.03,
            code_inactivity_timeout_seconds=0.08,
        )
        self.assertEqual(result, "finished")

    async def test_code_activity_uses_the_longer_silence_allowance(self):
        async def operation(progress_callback):
            await progress_callback(ProgressEvent(
                stage="MODEL_STARTED",
                message="Code Worker 正在执行",
                capability="code_agent",
                status="round-1",
            ))
            await asyncio.sleep(0.05)
            return "finished"

        result = await run_with_progress_keepalive(
            operation,
            AsyncMock(),
            interval_seconds=0.01,
            inactivity_timeout_seconds=0.03,
            code_inactivity_timeout_seconds=0.08,
        )
        self.assertEqual(result, "finished")

    async def test_selected_skill_is_reported_without_its_policy_body(self):
        events: list[ProgressEvent] = []

        async def capture(event: ProgressEvent) -> None:
            events.append(event)

        snapshot = SimpleNamespace(
            selected=(SimpleNamespace(name="web-cinema-showtimes"),),
            model_calls=0,
            model_dump=lambda **kwargs: {"selected": [{"name": "web-cinema-showtimes"}]},
        )
        middleware = RoleSkillsMiddleware(None, "web", ())
        with patch("skill_runtime.middleware.prepare_skills", new_callable=AsyncMock) as prepare:
            prepare.return_value = snapshot
            await middleware.abefore_agent(
                {"step_id": "3", "worker_id": "worker-a", "messages": []},
                None,
                {"configurable": {"progress_callback": capture}},
            )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].stage, "CAPABILITY_SELECTED")
        self.assertEqual(events[0].step_id, 3)
        self.assertIn("web-cinema-showtimes", events[0].message)

    def test_unknown_tool_name_is_bounded(self):
        label = tool_progress_label("private value / " + "x" * 200)
        self.assertNotIn("private value", label)
        self.assertLess(len(label), 80)

    async def test_parallel_updates_edit_one_feishu_message(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        function = next(
            node for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_progress_callback"
        )
        channel = SimpleNamespace(
            send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1")),
            edit_message=AsyncMock(return_value=SimpleNamespace(success=True)),
        )
        namespace = {
            "asyncio": asyncio,
            "feishu_channel": channel,
            "logger": logging.getLogger("test-live-progress"),
            "ProgressEvent": ProgressEvent,
            "render_progress_event": render_progress_event,
        }
        module = ast.Module(
            body=[ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        callback = namespace["_build_progress_callback"]("chat-1")
        await asyncio.gather(
            callback(ProgressEvent(stage="STEP_STARTED", message="第一步", step_id=1)),
            callback(ProgressEvent(stage="TOOL_STARTED", message="正在搜索", step_id=1, status="1")),
        )
        channel.send.assert_awaited_once()
        channel.edit_message.assert_awaited_once()
        self.assertEqual(channel.edit_message.await_args.args[0], "m1")

    async def test_progress_from_another_event_loop_returns_to_owner_loop(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        function = next(
            node for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_progress_callback"
        )
        channel = SimpleNamespace(
            send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1")),
            edit_message=AsyncMock(return_value=SimpleNamespace(success=True)),
        )
        namespace = {
            "asyncio": asyncio,
            "feishu_channel": channel,
            "logger": logging.getLogger("test-live-progress"),
            "ProgressEvent": ProgressEvent,
            "render_progress_event": render_progress_event,
        }
        module = ast.Module(
            body=[ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        callback = namespace["_build_progress_callback"]("chat-1")

        event = ProgressEvent(
            stage="MODEL_STARTED", message="跨循环进度", status="round-1",
        )
        await asyncio.to_thread(lambda: asyncio.run(callback(event)))
        channel.send.assert_awaited_once()

    async def test_progress_rotates_before_feishu_edit_limit(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        function = next(
            node for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_progress_callback"
        )
        channel = SimpleNamespace(
            send=AsyncMock(side_effect=[
                SimpleNamespace(success=True, message_id="m1"),
                SimpleNamespace(success=True, message_id="m2"),
            ]),
            edit_message=AsyncMock(return_value=SimpleNamespace(success=True)),
        )
        namespace = {
            "asyncio": asyncio,
            "feishu_channel": channel,
            "logger": logging.getLogger("test-live-progress"),
            "ProgressEvent": ProgressEvent,
            "render_progress_event": render_progress_event,
        }
        module = ast.Module(
            body=[ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        callback = namespace["_build_progress_callback"]("chat-1")

        for index in range(17):
            await callback(ProgressEvent(
                stage="MODEL_STARTED",
                message=f"进度 {index}",
                step_id=1,
                status=f"round-{index}",
            ))

        self.assertEqual(channel.send.await_count, 2)
        self.assertEqual(channel.edit_message.await_count, 15)
        self.assertEqual(channel.send.await_args_list[1].args[1]["text"], "进度续接：\n进度 16")

    async def test_progress_rotates_when_feishu_reports_edit_limit(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        function = next(
            node for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_progress_callback"
        )
        channel = SimpleNamespace(
            send=AsyncMock(side_effect=[
                SimpleNamespace(success=True, message_id="m1"),
                SimpleNamespace(success=True, message_id="m2"),
            ]),
            edit_message=AsyncMock(return_value=SimpleNamespace(
                success=False,
                error=SimpleNamespace(raw_code=230072),
            )),
        )
        namespace = {
            "asyncio": asyncio,
            "feishu_channel": channel,
            "logger": logging.getLogger("test-live-progress"),
            "ProgressEvent": ProgressEvent,
            "render_progress_event": render_progress_event,
        }
        module = ast.Module(
            body=[ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        callback = namespace["_build_progress_callback"]("chat-1")

        await callback(ProgressEvent(
            stage="STEP_STARTED", message="开始", step_id=1, status="attempt-1",
        ))
        await callback(ProgressEvent(
            stage="MODEL_STARTED", message="分析", step_id=1, status="round-1",
        ))

        self.assertEqual(channel.send.await_count, 2)
        self.assertEqual(channel.edit_message.await_count, 1)
        self.assertEqual(channel.send.await_args_list[1].args[1]["text"], "进度续接：\n分析")


if __name__ == "__main__":
    unittest.main()
