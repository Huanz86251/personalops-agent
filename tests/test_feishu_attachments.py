"""Offline ingress acceptance using actual SDK messages, SQLite, pump and OCR.

Only the network transport and paid model are replaced. Main adapter functions
are compiled from their real AST so importing main cannot start live services.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from lark_channel.channel.types import (
    Conversation,
    FileContent,
    Identity,
    ImageContent,
    InboundMessage,
    PostContent,
    ReplyRef,
    ResourceDescriptor,
    TextContent,
)
from test_local_ocr import image_bytes, pdf_bytes

from eventing import (
    AsyncEventStore,
    EventAction,
    EventConflictError,
    EventOrigin,
    EventRunPump,
    RunStatus,
    create_agent_event,
    create_event_run,
)
from feishu_attachments import AttachmentIngressError, FeishuAttachmentStore
from feishu_exports import ExportPolicy, ExportService, export_event_scope
from scripts import local_ocr_worker
from tools import local_native as local


class ScriptedReadingModel(BaseChatModel):
    """Emits a real LangChain tool call without contacting a model provider."""

    reading_path: str

    @property
    def _llm_type(self):
        return "offline-attachment-acceptance"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(content="Reading complete.")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "attachment_to_text",
                        "id": "real-reader",
                        "args": {
                            "path": self.reading_path,
                            "output_path": "/artifacts/read.md",
                        },
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def incoming(
    mid,
    text="",
    *,
    kind="text",
    key=None,
    name=None,
    sender="owner",
    chat="chat",
    reply=None,
    sources=None,
):
    content = {
        "text": TextContent,
        "post": PostContent,
        "image": ImageContent,
        "file": FileContent,
    }[kind]()
    return InboundMessage(
        id=mid,
        create_time=1,
        conversation=Conversation(chat),
        sender=Identity(sender),
        content=content,
        content_text=text,
        body_text=text,
        resources=[
            ResourceDescriptor(
                "image" if kind in {"image", "post"} else "file", key, name
            )
        ]
        if key
        else [],
        reply=ReplyRef(reply) if reply else None,
        batched_sources=sources,
    )


def adapter_namespace(store, runtime, channel, pump):
    source = Path(__file__).resolve().parents[1] / "main.py"
    wanted = {
        "_parse_command",
        "_enqueue_feishu_event",
        "_enqueue_feishu_replace",
        "_cancel_running_task",
        "_handle_pending_interaction",
        "_run_feishu_event",
        "handle_feishu_message",
    }
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    selected = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert len(selected) == len(wanted)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = {
        "asyncio": asyncio,
        "EventAction": EventAction,
        "EventOrigin": EventOrigin,
        "RunStatus": RunStatus,
        "EventConflictError": EventConflictError,
        "AttachmentIngressError": AttachmentIngressError,
        "create_agent_event": create_agent_event,
        "create_event_run": create_event_run,
        "OWNER_MESSAGE_LOCK": asyncio.Lock(),
        "FEISHU_ATTACHMENTS": store,
        "RAG_UPLOADS": __import__("knowledge_rag.ingress", fromlist=["RagUploadInbox"]).RagUploadInbox(store.root / "rag-mode"),
        "_send_control_panel": AsyncMock(),
        "FEISHU_EXPORTS": ExportService(store.root / "exports", ExportPolicy(), channel),
        "export_event_scope": export_event_scope,
        "FEISHU_EVENT_CONTEXTS": {},
        "FEISHU_INTERACTIONS": {},
        "conversation_runtime": runtime,
        "feishu_channel": channel,
        "FEISHU_CHANNEL": "feishu",
        "OWNER_EXTERNAL_CHAT_ID": "owner",
        "_require_event_pump": lambda: pump,
        "_send_text": AsyncMock(),
        "_build_progress_callback": lambda chat: AsyncMock(),
        "_compact_terminal_text": lambda text: text,
        "logger": logging.getLogger("ingress-test"),
        "_handle_command": AsyncMock(return_value=True),
        "_enqueue_feishu_cancel": AsyncMock(),
        "_handle_clean_action": AsyncMock(),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)  # noqa: S102 - only local checked-in function definitions
    return namespace


class FeishuIngressTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.append(str(local.OCR_DEPENDENCY_ROOT))
        cls.image = image_bytes()
        cls.pdf = pdf_bytes([None, cls.image])

    async def asyncSetUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = FeishuAttachmentStore(self.root / "attachments")
        self.channel = SimpleNamespace(
            download_resource=AsyncMock(return_value=self.image)
        )
        self.events = AsyncEventStore(self.root / "events.sqlite3")
        await self.events.start()
        self.addAsyncCleanup(self.events.close)
        self.runtime = SimpleNamespace(
            event_store=self.events,
            get_active_conversation=AsyncMock(
                return_value=SimpleNamespace(conversation_id="conv")
            ),
            ask=AsyncMock(return_value="answer"),
        )
        self.pump = SimpleNamespace(active_event_id=None, notify=Mock())
        self.ns = adapter_namespace(self.store, self.runtime, self.channel, self.pump)

    async def prepare(self, message, conversation="conv"):
        return await self.store.prepare(message, conversation, self.channel)

    async def test_file_only_then_reply_enqueues_once_and_preserves_original_message_id(
        self,
    ):
        uploaded = incoming("image-1", "![image](key1)", kind="image", key="key1")
        await self.ns["handle_feishu_message"](uploaded)
        self.pump.notify.assert_not_called()
        ack = self.ns["_send_text"].call_args.args[1]
        self.assertIn("已保存 1 个附件", ack)
        self.assertNotIn("![image]", ack)
        self.channel.download_resource.assert_awaited_once_with(
            "key1", resource_type="image", message_id="image-1"
        )
        followup = incoming("text-2", "请读取这张表", reply="image-1")
        await self.ns["handle_feishu_message"](followup)
        event = self.pump.notify.call_args.args[0]
        self.assertIn("/handoff/inbound/", event.payload_text)
        self.assertIn("ocr_image", event.payload_text)
        await self.ns["handle_feishu_message"](followup)
        self.pump.notify.assert_called_once()
        self.channel.download_resource.assert_awaited_once()

    async def test_actual_sdk_rich_post_cleans_placeholder_and_persists_task(self):
        message = incoming("post", "请读表\n![image](pic)", kind="post", key="pic")
        await self.ns["handle_feishu_message"](message)
        event = self.pump.notify.call_args.args[0]
        self.assertTrue(event.payload_text.startswith("请读表\n\n"))
        self.assertNotIn("![image]", event.payload_text)
        self.assertEqual(
            (await self.events.require_run(event.event_id)).status, RunStatus.QUEUED
        )

    async def test_same_name_different_files_and_explicit_reference_survive_restart(
        self,
    ):
        self.channel.download_resource.side_effect = [
            self.image,
            image_bytes("第二份 999", "Other 999"),
        ]
        first = await self.prepare(
            incoming("f1", kind="file", key="k1", name="same.bin")
        )
        second = await self.prepare(
            incoming("f2", kind="file", key="k2", name="same.bin")
        )
        self.assertNotEqual(first.attachments[0]["id"], second.attachments[0]["id"])
        self.assertNotEqual(
            first.attachments[0]["sha256"], second.attachments[0]["sha256"]
        )
        self.store = FeishuAttachmentStore(self.store.root)
        referenced = await self.prepare(
            incoming("use", f"读取{first.attachments[0]['id']}这个附件")
        )
        self.assertEqual(referenced.attachments, first.attachments)
        self.assertEqual(self.channel.download_resource.await_count, 2)

    async def test_reference_rejects_other_sender_chat_or_conversation(self):
        first = await self.prepare(
            incoming("private", kind="file", key="k", name="a.png")
        )
        aid = first.attachments[0]["id"]
        for overrides, conversation in [
            ({"sender": "other"}, "conv"),
            ({"chat": "elsewhere"}, "conv"),
            ({}, "otherconv"),
        ]:
            with self.assertRaises(AttachmentIngressError):
                await self.prepare(
                    incoming("ref", "读取 " + aid, **overrides), conversation
                )
        # A scoped reply cannot silently import someone else's resource either.
        with self.assertRaises(AttachmentIngressError):
            await self.prepare(
                incoming("foreign-reply", "读取", sender="other", reply="private")
            )

    async def test_conversation_switch_during_download_and_duplicate_retry_keep_binding(
        self,
    ):
        first = await self.prepare(
            incoming("same-message", "读取", kind="post", key="k")
        )
        self.runtime.get_active_conversation.return_value = SimpleNamespace(
            conversation_id="newconv"
        )
        event = await self.ns["_enqueue_feishu_event"](
            chat_id="chat", payload_text="读取", action=EventAction.QUEUE, inbound=first
        )
        self.assertEqual(event.conversation_id, "conv")
        duplicate = await self.prepare(
            incoming("same-message", "读取", kind="post", key="k"), "newconv"
        )
        self.assertEqual(duplicate.event_id, event.event_id)
        self.assertEqual(duplicate.conversation_id, "conv")
        self.channel.download_resource.assert_awaited_once()
        foreign = create_agent_event(
            conversation_id="newconv",
            action=EventAction.QUEUE,
            payload_text="unrelated",
            origin=EventOrigin.FEISHU,
        )
        await self.events.add_event(foreign, run=create_event_run(foreign))
        self.pump.active_event_id = foreign.event_id
        inserted = await self.prepare(
            incoming("insert-old", "读图", reply="same-message"), "conv"
        )
        inserted_event = await self.ns["_enqueue_feishu_event"](
            chat_id="chat",
            payload_text="读图",
            action=EventAction.INSERT,
            inbound=inserted,
        )
        self.assertEqual(inserted_event.conversation_id, "conv")
        self.assertIsNone(inserted_event.target_event_id)

    async def test_batch_uses_each_resource_original_message_id(self):
        parts = [
            incoming("orig-img", kind="image", key="image"),
            incoming("orig-text", "读图"),
        ]
        batched = incoming("batch", sources=parts)
        prepared = await self.prepare(batched)
        self.assertEqual(prepared.instruction, "读图")
        self.channel.download_resource.assert_awaited_once_with(
            "image", resource_type="image", message_id="orig-img"
        )

    async def test_download_failure_or_oversize_or_unknown_binary_never_enqueue(self):
        for i, data in enumerate([None, b"x" * 101, b"\0not a document"]):
            self.channel.download_resource.return_value = data
            with patch("feishu_attachments.MAX_FILE_BYTES", 100):
                await self.ns["handle_feishu_message"](
                    incoming(f"bad-{i}", "读", kind="post", key="k")
                )
        self.pump.notify.assert_not_called()
        self.runtime.ask.assert_not_awaited()
        self.assertTrue(
            all(
                "未进入任务队列" in call.args[1]
                for call in self.ns["_send_text"].call_args_list
            )
        )

    async def test_path_name_is_sanitized_and_saved_bytes_are_verified(self):
        prepared = await self.prepare(
            incoming("unsafe-name", "读", kind="post", key="k", name="../../CON.png")
        )
        event = await self.ns["_enqueue_feishu_event"](
            chat_id="chat",
            payload_text="读",
            action=EventAction.QUEUE,
            inbound=prepared,
        )
        paths = self.store.materialize(event, self.root / "runs")
        self.assertNotIn("../", paths[0])
        record = prepared.attachments[0]
        (self.store.root / "objects" / record["object_name"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(AttachmentIngressError, "损坏"):
            self.store.materialize(event, self.root / "runs")

    async def test_insert_command_carries_manifest_and_control_commands_stay_out_of_model(
        self,
    ):
        await self.ns["handle_feishu_message"](
            incoming("insert", "/insert 读取图片", kind="post", key="k")
        )
        event = self.pump.notify.call_args.args[0]
        self.assertEqual(event.action, EventAction.INSERT)
        self.assertTrue(event.payload_text.startswith("读取图片\n"))
        self.assertIn("/handoff/inbound/", event.payload_text)
        await self.ns["handle_feishu_message"](incoming("new", "/new 新对话"))
        self.ns["_handle_command"].assert_awaited_once_with("chat", "new", ["新对话"])
        self.pump.notify.assert_called_once()

    async def test_inspection_never_loads_ocr_model(self):
        source = self.root / "inspect.pdf"
        source.write_bytes(self.pdf)
        factory = Mock(side_effect=AssertionError("must not load OCR"))
        result = local_ocr_worker.run(
            {"input_path": str(source), "name": "renamed.bin", "operation": "inspect"},
            engine_factory=factory,
        )
        self.assertEqual(result["summary"]["format"], "pdf")
        factory.assert_not_called()

    async def test_concurrent_duplicate_messages_create_only_one_task(self):
        message = incoming("concurrent", "读图", kind="post", key="k")
        await asyncio.gather(
            *(self.ns["handle_feishu_message"](message) for _ in range(3))
        )
        self.channel.download_resource.assert_awaited_once()
        self.pump.notify.assert_called_once()

    async def test_replace_command_binds_attachments_to_new_event(self):
        target = create_agent_event(
            conversation_id="conv",
            action=EventAction.QUEUE,
            payload_text="old",
            reply_target_id="chat",
            origin=EventOrigin.FEISHU,
        )
        await self.events.add_event(target, run=create_event_run(target))
        await self.events.update_run_status(target.event_id, RunStatus.RUNNING)
        self.pump.active_event_id = target.event_id
        await self.ns["handle_feishu_message"](
            incoming("replace", "/replace 改为读图", kind="post", key="k")
        )
        event = self.pump.notify.call_args.args[0]
        self.assertEqual(event.action, EventAction.REPLACE)
        self.assertEqual(event.target_event_id, target.event_id)
        self.assertEqual(len(self.store.materialize(event, self.root / "runs")), 1)
        await self.ns["handle_feishu_message"](
            incoming("replace", "/replace 改为读图", kind="post", key="k")
        )
        self.pump.notify.assert_called_once()

    async def test_planning_step_keeps_attachment_manifest_in_worker_instruction(self):
        from dataclasses import fields

        from config import PlanningSettings
        from planning_graph import _build_step_instruction
        from planning_models import PlanningContextPack, PlanStep

        prepared = await self.prepare(incoming("plan", "读取", kind="post", key="k"))
        payload = prepared.payload()
        instruction = _build_step_instruction(
            {
                "context": PlanningContextPack(
                    current_time="2026-09-05", user_request=payload
                ),
                "plan_objective": "读取附件",
            },
            PlanStep(
                step_id=1,
                objective="提取附件文字",
                success_criteria=["返回文字"],
                worker_kind="GENERAL",
            ),
            1,
            model_limit=5,
            tool_limit=5,
            planning=PlanningSettings(
                **{field.name: 20 for field in fields(PlanningSettings)}
            ),
        )
        self.assertIn(payload, instruction)

    async def test_missing_input_ledger_stops_recovery_before_model_call(self):
        await self.ns["handle_feishu_message"](
            incoming("lost", "读图", kind="post", key="k")
        )
        event = self.pump.notify.call_args.args[0]
        self.ns["FEISHU_ATTACHMENTS"] = FeishuAttachmentStore(self.root / "empty")
        with self.assertRaisesRegex(AttachmentIngressError, "登记缺失"):
            await self.ns["_run_feishu_event"](event, None, True)
        self.runtime.ask.assert_not_awaited()

    async def test_control_commands_do_not_wait_for_attachment_downloads(self):
        with patch.object(
            self.store,
            "prepare",
            AsyncMock(side_effect=AssertionError("control must bypass download queue")),
        ):
            await self.ns["handle_feishu_message"](incoming("cancel-now", "/cancel"))
            await self.ns["handle_feishu_message"](incoming("new-now", "/new next"))
        self.ns["_enqueue_feishu_cancel"].assert_awaited_once_with(chat_id="chat")
        self.ns["_handle_command"].assert_awaited_once_with("chat", "new", ["next"])

    async def test_button_started_insert_consumes_next_plain_message_locally(self):
        self.ns["FEISHU_INTERACTIONS"]["chat"] = SimpleNamespace(
            mode="insert",
            token="token",
        )
        await self.ns["handle_feishu_message"](
            incoming("button-insert", "优先检查部署日志")
        )
        event = self.pump.notify.call_args.args[0]
        self.assertEqual(event.action, EventAction.INSERT)
        self.assertEqual(event.payload_text, "优先检查部署日志")
        self.assertNotIn("chat", self.ns["FEISHU_INTERACTIONS"])
        self.runtime.ask.assert_not_awaited()

    async def test_clean_mode_a_is_a_local_choice_not_a_model_message(self):
        self.ns["FEISHU_INTERACTIONS"]["chat"] = SimpleNamespace(
            mode="clean",
            token="token",
        )
        await self.ns["handle_feishu_message"](incoming("clean-a", "A"))
        self.ns["_handle_clean_action"].assert_awaited_once_with(
            "chat", "choose_a"
        )
        self.pump.notify.assert_not_called()
        self.runtime.ask.assert_not_awaited()

    async def test_durable_queue_recovery_runs_real_pdf_reader_and_releases_ocr(self):
        self.channel.download_resource.return_value = self.pdf
        upload = await self.prepare(
            incoming("pdf", kind="file", key="file-key", name="report.bin")
        )
        message = incoming("read-pdf", "读取所有页", reply="pdf")
        await self.ns["handle_feishu_message"](message)
        event = self.pump.notify.call_args.args[0]
        self.assertEqual(upload.attachments[0]["format"], "pdf")
        # Simulate process loss after event persistence, before pump.notify runs.
        await self.events.close()
        await self.events.start()
        restored = FeishuAttachmentStore(self.store.root)
        storage = self.root / "runs"
        self.ns["FEISHU_ATTACHMENTS"] = SimpleNamespace(
            materialize=lambda e: restored.materialize(e, storage),
            export_identity=restored.export_identity,
        )
        self.ns["FEISHU_EVENT_CONTEXTS"].clear()
        self.channel.download_resource.side_effect = AssertionError(
            "recovery must use saved bytes"
        )

        async def ask(**kwargs):
            from workers.general_runtime import GeneralStepRuntime

            current = await self.events.require_event(kwargs["event_id"])
            paths = restored.materialize(current, storage)
            worker = GeneralStepRuntime(
                ScriptedReadingModel(reading_path=paths[0]),
                event_store=self.events,
                tools=[local.attachment_to_text],
                progress_every_tool_calls=4,
                run_storage_root=storage,
            )
            with patch.dict(os.environ, {"SKILL_ROUTING_MODE": "off"}):
                state = await worker.ainvoke(
                    {
                        "event_id": current.event_id,
                        "worker_id": "test-feishu-reader",
                        "step_id": "1",
                        "messages": [{"role": "user", "content": kwargs["user_text"]}],
                    }
                )
            results = [
                m
                for m in state["messages"]
                if isinstance(m, ToolMessage) and m.tool_call_id == "real-reader"
            ]
            self.assertEqual(len(results), 1)
            result = json.loads(results[0].content)
            self.assertIn("/artifacts/read.md", state["files"])
            self.assertIn("Native heading", result["preview"])
            self.assertIn("中文测试", result["preview"])
            self.assertIn("English OCR Total 123.45", result["preview"])
            self.assertEqual(result["reading"]["engine_loads"], 1)
            self.assertTrue(result["reading"]["process_released"])
            import psutil

            self.assertFalse(psutil.pid_exists(result["reading"]["worker_pid"]))
            return result["preview"]

        self.runtime.ask.side_effect = ask
        done, failures = asyncio.Event(), []

        async def delivered(event, result):
            done.set()

        async def failed(event, error):
            failures.append(error)
            done.set()

        pump = EventRunPump(
            self.events,
            handler=self.ns["_run_feishu_event"],
            on_result=delivered,
            on_failure=failed,
        )
        try:
            pump.start()
            await asyncio.wait_for(done.wait(), 60)
            if failures:
                raise failures[0]
            self.assertEqual(
                (await self.events.require_run(event.event_id)).status,
                RunStatus.COMPLETED,
            )
            self.runtime.ask.assert_awaited_once()
        finally:
            await pump.stop()
