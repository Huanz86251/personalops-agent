import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from langchain_core.utils.function_calling import convert_to_openai_tool

from host_file_access import (
    HOST_FILE_ACCESS,
    HostFileAccessService,
    HostFileApprovalRequest,
    HostFileJudgeOutput,
    HostFilePolicy,
    HostFileUserDenied,
    configure_host_file_policy,
)
from tools.host_file_tools import (
    host_import_file,
    host_list_files,
    host_search_files,
)


class _StructuredJudge:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        return HostFileJudgeOutput(decision=self.decision, reason="offline fixture")


class _Judge:
    def __init__(self, decision):
        self.structured = _StructuredJudge(decision)

    def with_structured_output(self, _schema):
        return self.structured


class _BrokenStructuredJudge:
    async def ainvoke(self, _messages):
        raise ValueError("malformed judge output")


class _BrokenJudge:
    def with_structured_output(self, _schema):
        return _BrokenStructuredJudge()


def approval_request(path, *, operation="SEND_TO_USER"):
    return HostFileApprovalRequest(
        event_id="event-1",
        chat_id="chat-1",
        worker_id="worker-1",
        worker_kind="GENERAL",
        operation=operation,
        path=str(path),
        worker_reason="读取用户明确提到的简历",
        original_user_request="读取我电脑里的简历并给建议",
        current_step="找到并读取简历",
    )


class HostFileAccessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temp = TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.file = self.root / "resume.pdf"
        self.file.write_bytes(b"fixture")
        self.service = HostFileAccessService(HostFilePolicy((self.root,)))

    async def test_clear_judge_allow_does_not_interrupt_user(self):
        judge = _Judge("ALLOW")
        self.service.configure_judge(judge)
        notified = []

        async def notify(*args):
            notified.append(args)

        self.service.configure_human_notifier(notify)
        await self.service.authorize(approval_request(self.file))
        self.assertEqual(notified, [])
        self.assertEqual(judge.structured.calls, 1)

    async def test_overlong_judge_reason_is_deterministically_truncated(self):
        class _LongStructured:
            async def ainvoke(self, _messages):
                return {"decision": "ALLOW", "reason": "理由" * 400}

        class _LongJudge:
            def with_structured_output(self, _schema):
                return _LongStructured()

        self.service.configure_judge(_LongJudge())
        judgment = await self.service._judge(approval_request(self.file))
        self.assertEqual(judgment.decision, "ALLOW")
        self.assertEqual(len(judgment.reason), 300)
        self.assertTrue(judgment.reason.endswith("…"))

    async def test_judge_parse_failure_falls_back_to_same_user_confirmation(self):
        self.service.configure_judge(_BrokenJudge())
        notices = []

        async def notify(chat_id, request):
            notices.append((chat_id, request.operation, request.path))
            self.service.consume_user_reply(chat_id, "允许")

        self.service.configure_human_notifier(notify)
        await self.service.authorize(approval_request(self.file))
        self.assertEqual(
            notices,
            [("chat-1", "SEND_TO_USER", str(self.file))],
        )

    async def test_user_refusal_is_explicit_and_cached_as_denial(self):
        judge = _Judge("ASK_USER")
        self.service.configure_judge(judge)

        async def notify(chat_id, _request):
            reply = self.service.consume_user_reply(chat_id, "拒绝")
            self.assertTrue(reply.handled)
            self.assertIn("USER_DENIED", reply.reply)

        self.service.configure_human_notifier(notify)
        with self.assertRaisesRegex(HostFileUserDenied, "用户拒绝不允许使用该文件"):
            await self.service.authorize(approval_request(self.file))
        with self.assertRaisesRegex(HostFileUserDenied, "用户拒绝不允许使用该文件"):
            await self.service.authorize(approval_request(self.file))
        self.assertEqual(judge.structured.calls, 1)

    async def test_unknown_reply_keeps_request_pending_until_exact_decision(self):
        self.service.configure_judge(_Judge("ASK_USER"))

        async def notify(chat_id, _request):
            invalid = self.service.consume_user_reply(chat_id, "可以吧")
            self.assertIn("只回复", invalid.reply)
            asyncio.get_running_loop().call_soon(
                self.service.consume_user_reply, chat_id, "允许"
            )

        self.service.configure_human_notifier(notify)
        await self.service.authorize(approval_request(self.file))

    async def test_path_outside_root_is_rejected_even_before_model_policy(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("x", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        with self.assertRaisesRegex(PermissionError, "不在本机配置"):
            self.service.policy.validate(str(outside), require_file=True)

    async def test_configured_absolute_roots_are_embedded_in_tool_descriptions(self):
        original_policy = HOST_FILE_ACCESS.policy
        second_root = self.root / "extra-downloads"
        second_root.mkdir()
        try:
            configure_host_file_policy(HostFilePolicy((self.root, second_root)))
            expected_roots = (str(self.root.resolve()), str(second_root.resolve()))
            for tool in (host_list_files, host_search_files, host_import_file):
                for expected in expected_roots:
                    self.assertIn(expected, tool.description)
                self.assertIn("只读访问", tool.description)
                serialized_description = convert_to_openai_tool(tool)["function"][
                    "description"
                ]
                for expected in expected_roots:
                    self.assertIn(expected, serialized_description)
        finally:
            configure_host_file_policy(original_policy)

    async def test_external_mutation_always_asks_human_and_parent_covers_children(self):
        judge = _Judge("ALLOW")
        self.service.configure_judge(judge)
        folder = self.root / "approved-folder"
        folder.mkdir()
        child = folder / "child.txt"
        child.write_text("x", encoding="utf-8")
        notified = []

        async def notify(chat_id, request):
            notified.append(request.path)
            self.service.consume_user_reply(chat_id, "允许")

        self.service.configure_human_notifier(notify)
        await self.service.authorize(
            approval_request(folder, operation="WRITE_EXTERNAL")
        )
        await self.service.authorize(
            approval_request(child, operation="WRITE_EXTERNAL")
        )
        self.assertEqual(notified, [str(folder)])
        self.assertEqual(judge.structured.calls, 0)

        await self.service.authorize(
            approval_request(folder, operation="DELETE_EXTERNAL")
        )
        await self.service.authorize(
            approval_request(child, operation="DELETE_EXTERNAL")
        )
        self.assertEqual(notified, [str(folder), str(folder)])
        self.assertEqual(judge.structured.calls, 0)

    async def test_bounded_search_returns_shallow_match_when_later_tree_is_truncated(self):
        original_policy = HOST_FILE_ACCESS.policy
        try:
            configure_host_file_policy(HostFilePolicy((self.root,)))
            shallow = self.root / "我的简历.pdf"
            shallow.write_bytes(b"resume")
            deep = self.root / "archive"
            deep.mkdir()
            for index in range(120):
                (deep / f"noise-{index:03d}.txt").write_text("x", encoding="utf-8")
            result = await host_search_files.coroutine(
                str(self.root),
                "简历",
                SimpleNamespace(state={}, tool_call_id="search-1"),
                scan_limit=100,
            )
            self.assertTrue(result["truncated"])
            self.assertEqual(result["scanned_entries"], 100)
            self.assertEqual([item["path"] for item in result["items"]], [str(shallow.resolve())])
        finally:
            configure_host_file_policy(original_policy)

    async def test_large_directory_listing_returns_a_truncated_subset(self):
        original_policy = HOST_FILE_ACCESS.policy
        try:
            configure_host_file_policy(HostFilePolicy((self.root,)))
            for index in range(120):
                (self.root / f"item-{index:03d}.txt").write_text("x", encoding="utf-8")
            result = await host_list_files.coroutine(
                str(self.root),
                SimpleNamespace(state={}, tool_call_id="list-1"),
                page_size=25,
                scan_limit=100,
            )
            self.assertTrue(result["truncated"])
            self.assertEqual(result["scanned_entries"], 100)
            self.assertEqual(result["returned"], 25)
            self.assertEqual(result["total"], 100)
            self.assertIsNotNone(result["next_cursor"])
        finally:
            configure_host_file_policy(original_policy)

    async def test_import_is_read_only_and_does_not_call_judge(self):
        original_policy = HOST_FILE_ACCESS.policy
        original_judge = HOST_FILE_ACCESS.judge_model
        judge = _Judge("ASK_USER")
        try:
            configure_host_file_policy(HostFilePolicy((self.root,)))
            HOST_FILE_ACCESS.configure_judge(judge)
            state = {
                "event_id": "event-read-without-judge",
                "worker_id": "general-read-without-judge",
            }
            command = await host_import_file.coroutine(
                str(self.file),
                "读取用户指定的简历",
                SimpleNamespace(state=state, tool_call_id="import-1"),
            )
            update = command.update
            self.assertEqual(len(update["worker_downloaded_artifacts"]), 1)
            self.assertEqual(judge.structured.calls, 0)
        finally:
            HOST_FILE_ACCESS.configure_judge(original_judge)
            configure_host_file_policy(original_policy)


if __name__ == "__main__":
    unittest.main()
