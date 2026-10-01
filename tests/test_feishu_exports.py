"""Offline authorization, snapshot, delivery and real General tool-call tests."""

import asyncio
import json
import os
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from lark_channel.channel.types import SendResult

from feishu_exports import (
    ExportDenied,
    ExportIdentity,
    ExportPolicy,
    ExportService,
    ExportUncertain,
    PLATFORM_UPLOAD_LIMIT,
    configure_exports,
    export_event_scope,
    export_general_scope,
    export_worker_scope,
    export_tool_available,
    request_export,
)
from tools.feishu_file_tools import send_local_file_to_feishu
from tools.host_file_tools import host_import_file
from host_file_access import (
    HOST_FILE_ACCESS,
    HostFileJudgeOutput,
    HostFilePolicy,
    configure_host_file_policy,
)
from run_workspace import initialize_run_workspace
from task_files import register_file
from workers.general_worker import select_general_worker_tools
from workers.web_worker import select_web_worker_tools


class ExportModel(BaseChatModel):
    path: str
    expect_oversize: bool = False

    @property
    def _llm_type(self):
        return "offline-export-test"

    def bind_tools(self, tools, **kwargs):
        assert "send_local_file_to_feishu" in {t.name for t in tools}
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            call = {
                "id": "export-test",
                "name": "send_local_file_to_feishu",
                "args": {
                    "source_refs": ["U1"],
                    "path": self.path,
                    "reason": "用户明确要求把这个文件发回当前会话。",
                },
            }
        else:
            content = results[-1].content
            if content.startswith("[evidence_ref: "):
                content = content.split("\n", 1)[1]
            if self.expect_oversize:
                assert "超过单次发送上限" in content, content
                status = "BLOCKED"
                summary = "文件超过发送上限，未上传；请用户确认是否允许拆分。"
                unresolved_items = ["等待用户决定是否拆分后分别发送"]
            else:
                assert content.startswith("{"), content
                payload = json.loads(content)
                assert payload["status"] == "SENT", payload
                assert payload["message_id"] == "om_receipt", payload
                status = "COMPLETED"
                summary = "文件已发送到当前飞书会话。"
                unresolved_items = []
            call = {
                "id": "report-test",
                "name": "report_general_result",
                "args": {
                    "result": {
                        "status": status,
                        "summary": summary,
                        "unresolved_items": unresolved_items,
                        "evidence_tool_call_ids": ["E1"],
                    }
                },
            }
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="", tool_calls=[call]))
            ]
        )


class FeishuExportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.allowed = self.root / "allowed"
        self.allowed.mkdir()
        self.file = self.allowed / "资料.pdf"
        self.file.write_bytes(b"pdf original")
        self.channel = SimpleNamespace(
            send=AsyncMock(return_value=SendResult.ok("om_receipt")),
            upload_media=AsyncMock(return_value="file_key"),
        )
        self.policy = ExportPolicy(
            frozenset({"owner"}), frozenset({"chat"}), (self.allowed,)
        )
        self.service = ExportService(self.root / "exports", self.policy, self.channel)
        self.identity = ExportIdentity("evt_export", "conversation", "owner", "chat")
        configure_exports(self.service)
        self.addCleanup(configure_exports, None)
        class _AllowStructured:
            async def ainvoke(self, _messages):
                return HostFileJudgeOutput(
                    decision="ALLOW", reason="测试中用户明确要求发送"
                )
        class _AllowJudge:
            def with_structured_output(self, _schema):
                return _AllowStructured()
        HOST_FILE_ACCESS.configure_judge(_AllowJudge())
        self.addCleanup(HOST_FILE_ACCESS.configure_judge, None)

    async def prepare(self, path=None):
        return await self.service.request(self.identity, str(path or self.file))

    def stage(self, path=None):
        return self.service._view(
            self.service._prepare(self.identity, str(path or self.file))
        )

    async def test_authorized_request_sends_without_approval_using_snapshot(self):
        from lark_channel.channel.outbound.media.uploader import gather_buffer

        uploaded = []

        async def upload(source, **kwargs):
            self.file.write_bytes(b"changed after snapshot")
            data, actual_name = await gather_buffer(
                source, default_name=kwargs["file_name"]
            )
            uploaded.append((source.kind, data, actual_name))
            return "file_123"

        self.channel.upload_media.side_effect = upload
        sent = await self.prepare()
        self.assertEqual(uploaded, [("buffer", b"pdf original", "资料.pdf")])
        self.assertEqual(sent["status"], "SENT")
        self.assertEqual(sent["message_id"], "om_receipt")
        self.assertEqual(self.channel.send.call_args.args[0], "chat")
        self.assertEqual(self.channel.send.call_args.args[1].source.key, "file_123")
        self.channel.send.assert_awaited_once()

    async def test_wrong_owner_chat_role_event_and_missing_configuration_fail_closed(
        self,
    ):
        for sender, chat in [("stranger", "chat"), ("owner", "other")]:
            identity = ExportIdentity("evt_export", "conversation", sender, chat)
            with self.assertRaises(ExportDenied):
                await self.service.request(identity, str(self.file))
        self.assertFalse(export_tool_available())
        with export_event_scope(self.identity):
            with self.assertRaises(ExportDenied):
                await request_export(str(self.file), self.identity.event_id)
            with export_general_scope(), self.assertRaises(ExportDenied):
                await request_export(str(self.file), "forged-event")
        empty = ExportService(self.root / "empty", ExportPolicy(), self.channel)
        with self.assertRaises(ExportDenied):
            await empty.request(self.identity, str(self.file))
        self.channel.send.assert_not_called()
        self.channel.upload_media.assert_not_called()

    async def test_other_owner_cannot_approve_or_query_request_in_shared_chat(self):
        result = self.stage()
        self.service.policy = ExportPolicy(
            frozenset({"owner", "owner2"}),
            frozenset({"chat", "chat2"}),
            (self.allowed,),
        )
        for sender, chat in [("owner2", "chat"), ("owner", "chat2")]:
            with self.assertRaises(ExportDenied):
                await self.service.decide(
                    result["request_id"], sender, chat, approve=True
                )
        self.channel.upload_media.assert_not_called()

    async def test_duplicate_direct_request_and_restart_never_send_twice(self):
        results = await asyncio.gather(
            *[self.prepare() for _ in range(2)]
        )
        self.assertTrue(all(r["status"] == "SENT" for r in results))
        restarted = ExportService(self.service.root, self.policy, self.channel)
        again = await restarted.request(self.identity, str(self.file))
        self.assertEqual(again["status"], "SENT")
        self.channel.upload_media.assert_awaited_once()

    async def test_ambiguous_delivery_remains_unknown_and_is_not_retried(self):
        self.channel.send.side_effect = TimeoutError("response lost")
        with self.assertRaises(ExportUncertain):
            await self.prepare()
        again = await self.prepare()
        self.assertEqual(again["status"], "UNKNOWN")
        self.channel.upload_media.assert_awaited_once()

    async def test_crash_claim_is_not_retried(self):
        result = self.stage()
        self.service._query(
            "UPDATE exports SET status='SENDING' WHERE id=?", (result["request_id"],)
        )
        again = await self.prepare()
        self.assertEqual(again["status"], "SENDING")
        self.channel.upload_media.assert_not_called()

    async def test_any_local_file_is_sendable_after_judge_but_unsafe_links_are_rejected(self):
        outside = self.root / "private.txt"
        outside.write_text("secret")
        secret = self.allowed / ".env"
        secret.write_text("secret")
        for path in [outside, secret]:
            sent = await self.prepare(path)
            self.assertEqual(sent["status"], "SENT")
        with self.assertRaises(ExportDenied):
            await self.prepare(Path("relative.txt"))
        hard = self.allowed / "hard.txt"
        os.link(outside, hard)
        with self.assertRaises(ExportDenied):
            await self.prepare(hard)
        link = self.allowed / "link.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            pass  # Windows may deny creating symlinks; hardlink check still ran.
        else:
            with self.assertRaises(ExportDenied):
                await self.prepare(link)
        self.assertEqual(self.channel.upload_media.await_count, 2)

    async def test_oversize_and_empty_rejected_before_upload(self):
        empty = self.allowed / "empty"
        empty.touch()
        with self.assertRaises(ExportDenied):
            await self.prepare(empty)
        self.service.policy = replace(self.policy, max_upload_bytes=4)
        with self.assertRaises(ExportDenied) as caught:
            await self.prepare()
        self.assertIn("超过单次发送上限", str(caught.exception))
        self.assertIn("4 字节，本机配置上限", str(caught.exception))
        self.assertIn("未上传", str(caught.exception))
        self.assertIn("询问用户是否允许拆分", str(caught.exception))
        self.channel.upload_media.assert_not_called()

    async def test_directory_zip_manifest_and_limits(self):
        folder = self.allowed / "reports"
        folder.mkdir()
        (folder / "一.txt").write_text("one")
        (folder / "two.txt").write_text("two")
        result = self.stage(folder)
        row = self.service._query(
            "SELECT * FROM exports WHERE id=?", (result["request_id"],)
        )[0]
        self.assertTrue(row["filename"].endswith(".zip"))
        with zipfile.ZipFile(
            self.service.root / "objects" / row["snapshot"]
        ) as archive:
            self.assertEqual(set(archive.namelist()), {"一.txt", "two.txt"})
        with patch("feishu_exports.MAX_FILES", 1), self.assertRaises(ExportDenied):
            await self.prepare(folder)
        (folder / ".env").write_text("secret")
        sent = await self.prepare(folder)
        self.assertEqual(sent["status"], "SENT")

    async def test_oversize_directory_archive_is_blocked_before_upload(self):
        folder = self.allowed / "archive"
        folder.mkdir()
        (folder / "random.bin").write_bytes(os.urandom(2048))
        self.service.policy = replace(self.policy, max_upload_bytes=100)
        with self.assertRaises(ExportDenied) as caught:
            await self.prepare(folder)
        self.assertIn("超过单次发送上限", str(caught.exception))
        self.assertIn("100 字节，本机配置上限", str(caught.exception))
        self.assertIn("未上传", str(caught.exception))
        self.channel.upload_media.assert_not_called()

    async def test_500_mb_local_setting_cannot_override_platform_limit(self):
        self.service.policy = replace(self.policy, max_upload_bytes=500_000_000)
        with self.file.open("wb") as output:
            output.truncate(PLATFORM_UPLOAD_LIMIT + 1)
        with self.assertRaises(ExportDenied) as caught:
            await self.prepare()
        self.assertIn("30 MB", str(caught.exception))
        self.assertIn("飞书平台上限", str(caught.exception))
        self.channel.upload_media.assert_not_called()

    async def test_reject_expire_and_tamper_do_not_upload(self):
        result = self.stage()
        rejected = await self.service.decide(
            result["request_id"], "owner", "chat", approve=False
        )
        self.assertEqual(rejected["status"], "REJECTED")
        self.file.write_bytes(b"second")
        second = self.stage()
        self.service._query(
            "UPDATE exports SET expires=0 WHERE id=?", (second["request_id"],)
        )
        expired = await self.service.decide(
            second["request_id"], "owner", "chat", approve=True
        )
        self.assertEqual(expired["status"], "EXPIRED")
        self.file.write_bytes(b"third")
        third = self.stage()
        row = self.service._query(
            "SELECT * FROM exports WHERE id=?", (third["request_id"],)
        )[0]
        (self.service.root / "objects" / row["snapshot"]).write_bytes(b"other")
        with self.assertRaises(ExportDenied):
            await self.service.decide(
                third["request_id"], "owner", "chat", approve=True
            )
        self.channel.upload_media.assert_not_called()

    async def test_schema_has_no_approval_or_recipient_and_all_workers_can_select_tool(
        self,
    ):
        schema = convert_to_openai_tool(send_local_file_to_feishu)["function"][
            "parameters"
        ]
        self.assertEqual(set(schema["properties"]), {"path", "reason"})
        self.assertEqual(select_general_worker_tools([send_local_file_to_feishu]), [])
        with export_event_scope(self.identity):
            self.assertEqual(
                select_general_worker_tools([send_local_file_to_feishu]), []
            )
            self.assertEqual(
                select_general_worker_tools(
                    [send_local_file_to_feishu], allow_file_export=True
                ),
                [send_local_file_to_feishu],
            )
            web_search = SimpleNamespace(name="web_search")
            self.assertIn(
                send_local_file_to_feishu,
                select_web_worker_tools([web_search, send_local_file_to_feishu]),
            )

    async def test_actual_general_runtime_tool_call_sends_without_owner_command(self):
        from eventing import AsyncEventStore
        from workers.general_runtime import GeneralStepRuntime

        events = AsyncEventStore(self.root / "events.sqlite3")
        await events.start()
        try:
            runtime = GeneralStepRuntime(
                ExportModel(path=str(self.file)),
                event_store=events,
                tools=[send_local_file_to_feishu],
                progress_every_tool_calls=4,
                run_storage_root=self.root / "runs",
            )
            with (
                patch.dict(os.environ, {"SKILL_ROUTING_MODE": "off"}),
                export_event_scope(self.identity),
            ):
                result = await runtime.ainvoke(
                    {
                        "messages": [{"role": "user", "content": "把这个文件发给我"}],
                        "worker_id": "general_export",
                        "event_id": self.identity.event_id,
                        "planning_run_id": self.identity.event_id,
                        "host_original_user_request": "把这个文件发给我",
                        "host_current_step": "把指定文件发送到当前飞书会话",
                        "host_worker_kind": "GENERAL",
                    }
                )
            self.assertEqual(result["general_result"]["status"], "COMPLETED")
            self.channel.upload_media.assert_awaited_once()
        finally:
            await events.close()

    async def test_general_imported_download_candidate_can_be_sent(self):
        captured = []

        class _CaptureStructured:
            async def ainvoke(self, messages):
                captured.extend(messages)
                return HostFileJudgeOutput(
                    decision="ALLOW", reason="可信映射与用户发送意图一致"
                )

        class _CaptureJudge:
            def with_structured_output(self, _schema):
                return _CaptureStructured()

        HOST_FILE_ACCESS.configure_judge(_CaptureJudge())
        original_policy = HOST_FILE_ACCESS.policy
        configure_host_file_policy(HostFilePolicy((self.root,)))
        self.addCleanup(configure_host_file_policy, original_policy)
        state = {
            "event_id": self.identity.event_id,
            "planning_run_id": self.identity.event_id,
            "worker_id": "general-import-send",
            "host_worker_kind": "GENERAL",
            "host_original_user_request": "把Downloads里的简历发给我",
            "host_current_step": "找到简历并发送到当前飞书会话",
            "run_storage_root": str(self.root / "runs"),
        }
        imported = await host_import_file.coroutine(
            str(self.file),
            "用户要求取得这份简历",
            SimpleNamespace(state=state, tool_call_id="import-general"),
        )
        state.update(imported.update)
        candidate_id = state["worker_downloaded_artifacts"][0]["candidate_id"]
        with export_event_scope(self.identity), export_worker_scope("general"):
            sent = await send_local_file_to_feishu.coroutine(
                f"/downloads/{candidate_id}",
                "用户明确要求把这份简历发回当前会话",
                SimpleNamespace(state=state, tool_call_id="send-general"),
            )
        self.assertEqual(sent["status"], "SENT")
        self.channel.upload_media.assert_awaited_once()
        judge_prompt = "\n".join(str(message.content) for message in captured)
        self.assertIn(str(self.file.resolve()), judge_prompt)
        self.assertIn(f"/downloads/{candidate_id}", judge_prompt)
        self.assertIn("已校验隔离副本", judge_prompt)

    async def test_uncertain_send_pauses_for_user_then_resumes_same_tool_call(self):
        class _AskStructured:
            async def ainvoke(self, _messages):
                return HostFileJudgeOutput(
                    decision="ASK_USER", reason="原始请求中的发送意图不够明确"
                )

        class _AskJudge:
            def with_structured_output(self, _schema):
                return _AskStructured()

        original_notifier = HOST_FILE_ACCESS.human_notifier
        HOST_FILE_ACCESS.configure_judge(_AskJudge())
        notices = []

        async def notify(chat_id, request):
            notices.append((chat_id, request.path, request.operation))
            reply = HOST_FILE_ACCESS.consume_user_reply(chat_id, "允许")
            self.assertTrue(reply.handled)

        HOST_FILE_ACCESS.configure_human_notifier(notify)
        self.addCleanup(HOST_FILE_ACCESS.configure_human_notifier, original_notifier)
        state = {
            "event_id": self.identity.event_id,
            "planning_run_id": self.identity.event_id,
            "worker_id": "general-user-approved-send",
            "host_worker_kind": "GENERAL",
            "host_original_user_request": "处理一下这个文件",
            "host_current_step": "如用户允许则发送文件",
        }
        with export_event_scope(self.identity), export_worker_scope("general"):
            sent = await send_local_file_to_feishu.coroutine(
                str(self.file),
                "希望把处理对象发回当前会话",
                SimpleNamespace(state=state, tool_call_id="send-after-user-approval"),
            )
        self.assertEqual(
            notices,
            [("chat", str(self.file), "SEND_TO_USER")],
        )
        self.assertEqual(sent["status"], "SENT")
        self.channel.upload_media.assert_awaited_once()

    async def test_web_download_candidate_can_be_sent(self):
        source = self.root / "web-result.pdf"
        source.write_bytes(b"downloaded web file")
        state = {
            "event_id": self.identity.event_id,
            "planning_run_id": self.identity.event_id,
            "worker_id": "web-download-send",
            "host_worker_kind": "WEB",
            "host_original_user_request": "从网上下载报告并发给我",
            "host_current_step": "下载报告并发送到当前飞书会话",
            "run_storage_root": str(self.root / "runs"),
        }
        record = register_file(
            source,
            source_root=self.root,
            state=state,
            tool_call_id="download-web",
            origin="WEB",
            source_ref="https://example.test/report.pdf",
        )
        state["worker_downloaded_artifacts"] = [record.model_dump(mode="json")]
        with export_event_scope(self.identity), export_worker_scope("web"):
            sent = await send_local_file_to_feishu.coroutine(
                f"/downloads/{record.candidate_id}",
                "用户明确要求把下载完成的报告发回当前会话",
                SimpleNamespace(state=state, tool_call_id="send-web"),
            )
        self.assertEqual(sent["status"], "SENT")
        self.channel.upload_media.assert_awaited_once()

    async def test_code_reviewed_handoff_can_be_sent(self):
        run_root = self.root / "runs"
        layout = initialize_run_workspace(
            self.identity.event_id,
            storage_root=run_root,
            canonical_root=self.root / "canonical",
        )
        reviewed = layout.handoff_root / "code" / "report.pdf"
        reviewed.parent.mkdir(parents=True, exist_ok=True)
        reviewed.write_bytes(b"reviewed code artifact")
        state = {
            "event_id": self.identity.event_id,
            "planning_run_id": self.identity.event_id,
            "worker_id": "code-handoff-send",
            "host_worker_kind": "CODE",
            "host_original_user_request": "生成PDF并发给我",
            "host_current_step": "发送Code Reviewer已批准的PDF",
            "run_storage_root": str(run_root),
        }
        with export_event_scope(self.identity), export_worker_scope("code"):
            sent = await send_local_file_to_feishu.coroutine(
                "/handoff/code/report.pdf",
                "用户明确要求接收已审核通过的PDF",
                SimpleNamespace(state=state, tool_call_id="send-code"),
            )
        self.assertEqual(sent["status"], "SENT")
        self.channel.upload_media.assert_awaited_once()

    async def test_worker_workspace_file_can_be_frozen_and_sent(self):
        state = {
            "event_id": self.identity.event_id,
            "planning_run_id": self.identity.event_id,
            "worker_id": "general-workspace-send",
            "host_worker_kind": "GENERAL",
            "host_original_user_request": "写一份说明并发给我",
            "host_current_step": "发送刚生成的说明文件",
            "run_storage_root": str(self.root / "runs"),
            "files": {
                "/artifacts/note.txt": {
                    "content": "workspace result",
                    "encoding": "utf-8",
                }
            },
        }
        with export_event_scope(self.identity), export_worker_scope("general"):
            sent = await send_local_file_to_feishu.coroutine(
                "/artifacts/note.txt",
                "用户明确要求接收刚生成的说明",
                SimpleNamespace(state=state, tool_call_id="send-workspace"),
            )
        self.assertEqual(sent["status"], "SENT")
        self.channel.upload_media.assert_awaited_once()

    async def test_oversize_error_reaches_general_and_asks_before_splitting(self):
        from eventing import AsyncEventStore
        from workers.general_runtime import GeneralStepRuntime

        events = AsyncEventStore(self.root / "events.sqlite3")
        await events.start()
        try:
            runtime = GeneralStepRuntime(
                ExportModel(path=str(self.file), expect_oversize=True),
                event_store=events,
                tools=[send_local_file_to_feishu],
                progress_every_tool_calls=4,
                run_storage_root=self.root / "runs",
            )
            self.service.policy = replace(self.policy, max_upload_bytes=4)
            with (
                patch.dict(os.environ, {"SKILL_ROUTING_MODE": "off"}),
                export_event_scope(self.identity),
            ):
                result = await runtime.ainvoke(
                    {
                        "messages": [{"role": "user", "content": "把这个文件发给我"}],
                        "worker_id": "general_export_oversize",
                        "event_id": self.identity.event_id,
                        "planning_run_id": self.identity.event_id,
                        "host_original_user_request": "把这个文件发给我",
                        "host_current_step": "把指定文件发送到当前飞书会话",
                        "host_worker_kind": "GENERAL",
                    }
                )
            self.assertEqual(result["general_result"]["status"], "PARTIAL")
            self.channel.upload_media.assert_not_called()
            self.channel.send.assert_not_called()
        finally:
            await events.close()

    async def test_control_rejects_forwarded_nontext_and_foreign_approvals(self):
        from test_feishu_attachments import incoming

        result = self.stage()
        text = f"/file_approve {result['request_id']}"
        for message in [
            incoming("foreign", text, sender="stranger"),
            incoming("post", text, sender="owner", kind="post"),
            incoming(
                "batch", text, sender="owner", sources=[incoming("original", text)]
            ),
        ]:
            self.assertTrue(await self.service.handle_control(message, text))
        self.channel.upload_media.assert_not_called()
        self.assertFalse(
            self.service.admit_message(
                incoming("foreign2", "read history", sender="stranger")
            )
        )
        self.assertTrue(
            self.service.admit_message(
                incoming("owner2", "read history", sender="owner")
            )
        )

    async def test_config_dummy_interface_and_tool_schema_forbid_forged_approval(self):
        from pydantic import ValidationError

        from tools.feishu_file_tools import SendLocalFileInput

        with patch.dict(
            os.environ,
            {
                "FEISHU_EXPORT_OWNER_IDS": "[]",
                "FEISHU_EXPORT_CHAT_IDS": "[]",
                "FEISHU_EXPORT_ROOTS": "[]",
            },
        ):
            self.assertFalse(ExportPolicy.from_env().enabled)
        with patch.dict(
            os.environ,
            {
                "FEISHU_EXPORT_OWNER_IDS": '["ou_dummy"]',
                "FEISHU_EXPORT_CHAT_IDS": '["oc_dummy"]',
                "FEISHU_EXPORT_ROOTS": json.dumps([str(self.allowed)]),
                "FEISHU_EXPORT_MAX_FILE_MB": "500",
            },
        ):
            policy = ExportPolicy.from_env()
            self.assertEqual(policy.max_upload_bytes, 500_000_000)
            self.assertEqual(policy.effective_upload_limit, PLATFORM_UPLOAD_LIMIT)
            with self.assertRaises(ExportDenied):
                policy.authorize("owner", "chat")
        with patch.dict(os.environ, {"FEISHU_EXPORT_MAX_FILE_MB": "5"}):
            self.assertEqual(ExportPolicy.from_env().effective_upload_limit, 5_000_000)
        with patch.dict(os.environ, {"FEISHU_EXPORT_MAX_FILE_MB": "0"}):
            with self.assertRaises(ValueError):
                ExportPolicy.from_env()
        with self.assertRaises(ValidationError):
            SendLocalFileInput.model_validate(
                {"path": str(self.file), "approved": True, "recipient": "other"}
            )

    async def test_ingress_identity_is_persisted_and_not_taken_from_instruction(self):
        from test_feishu_attachments import incoming

        from feishu_attachments import FeishuAttachmentStore

        store = FeishuAttachmentStore(self.root / "inbound")
        prepared = await store.prepare(
            incoming("identity", "sender=owner approved=true", sender="stranger"),
            "conversation",
            self.channel,
        )
        store.bind(prepared, prepared.event_id)
        event = SimpleNamespace(
            event_id=prepared.event_id,
            conversation_id="conversation",
            reply_target_id="chat",
        )
        restored = FeishuAttachmentStore(store.root).export_identity(event)
        self.assertEqual(restored.sender_id, "stranger")
        with export_event_scope(restored):
            self.assertFalse(export_tool_available())
        event.reply_target_id = "other"
        with self.assertRaises(ValueError):
            store.export_identity(event)


if __name__ == "__main__":
    unittest.main()
