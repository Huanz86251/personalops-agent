import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from config import EmailMCPSettings
from mcp_runtime import EmailMCPRuntime
from toolsets import DEFAULT_TOOLSET_REGISTRY


class _EmptyInput(BaseModel):
    pass


def _fake_tool(name: str) -> StructuredTool:
    async def invoke() -> str:
        return name

    return StructuredTool.from_function(
        coroutine=invoke,
        name=name,
        description=f"fake {name}",
        args_schema=_EmptyInput,
        infer_schema=False,
    )


class _SessionContext:
    def __init__(self) -> None:
        self.exited = False

    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited = True


class _FakeClient:
    created_configs: list[dict] = []
    contexts: list[_SessionContext] = []

    def __init__(self, config: dict) -> None:
        self.created_configs.append(config)

    def session(self, name: str) -> _SessionContext:
        if name != "email":
            raise AssertionError(name)
        context = _SessionContext()
        self.contexts.append(context)
        return context


def _settings(*, enabled: bool = True) -> EmailMCPSettings:
    return EmailMCPSettings(
        enabled=enabled,
        address="reader@example.com",
        auth_code="test-auth-code",
        imap_host="imap.example.com",
        imap_port=993,
        folder="INBOX",
        drafts_folder="Drafts",
        secure=True,
    )


class EmailMCPRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_download_root_is_account_and_session_private(self):
        first, second = EmailMCPRuntime(_settings()), EmailMCPRuntime(_settings())
        self.assertNotEqual(first.attachment_root, second.attachment_root)
        tool = first._wrap_tool(_fake_tool("qqmail_download_attachment"))
        self.assertEqual(tool.metadata["task_file_source_root"], str(first.attachment_root.resolve()))
        self.assertEqual(tool.metadata["task_file_origin"], "EMAIL")

    def setUp(self) -> None:
        _FakeClient.created_configs.clear()
        _FakeClient.contexts.clear()

    async def test_disabled_runtime_does_not_start_process(self) -> None:
        runtime = EmailMCPRuntime(_settings(enabled=False))
        with patch("mcp_runtime.MultiServerMCPClient") as client:
            await runtime.start()
        client.assert_not_called()
        self.assertEqual(runtime.tools, [])

    async def test_enabled_but_unconfigured_runtime_stays_disconnected(self) -> None:
        settings = EmailMCPSettings(
            enabled=True,
            address="",
            auth_code="",
            imap_host="imap.qq.com",
            imap_port=993,
            folder="INBOX",
            drafts_folder="Drafts",
            secure=True,
        )
        runtime = EmailMCPRuntime(settings)
        with patch("mcp_runtime.MultiServerMCPClient") as client:
            await runtime.start()
        client.assert_not_called()
        self.assertFalse(runtime.is_started)

    async def test_host_whitelist_allows_read_and_download_but_not_send(self) -> None:
        source_names = (
            "qqmail_connection_status",
            "qqmail_list_new_messages",
            "qqmail_get_snippet",
            "qqmail_get_message",
            "qqmail_list_attachments",
            "qqmail_download_attachment",
        )
        upstream_tools = [_fake_tool(name) for name in source_names]
        upstream_tools.append(_fake_tool("mail_send"))
        runtime = EmailMCPRuntime(_settings())

        with (
            patch("mcp_runtime.MultiServerMCPClient", _FakeClient),
            patch(
                "mcp_runtime.load_mcp_tools",
                AsyncMock(return_value=upstream_tools),
            ),
            patch.object(runtime, "_find_npx_command", return_value="npx.cmd"),
        ):
            await runtime.start()

        tool_names = {tool.name for tool in runtime.tools}
        self.assertEqual(
            tool_names,
            {
                "email_connection_status",
                "email_list_recent",
                "email_get_snippet",
                "email_read_message",
                "email_list_attachments",
                "email_download_attachment",
                "email_create_draft",
            },
        )
        self.assertNotIn("mail_send", tool_names)

        config = _FakeClient.created_configs[0]["email"]
        self.assertEqual(config["args"][-1], "@ethanli666/qqmail-mcp@1.2.1")
        self.assertEqual(config["env"]["QQMAIL_IMAP_HOST"], "imap.example.com")
        self.assertEqual(config["env"]["QQMAIL_PASS"], "test-auth-code")
        self.assertNotIn("DEEPSEEK_API_KEY", config["env"])
        self.assertNotIn("OPENAI_API_KEY", config["env"])
        self.assertNotIn("FEISHU_APP_SECRET", config["env"])

        resolution = DEFAULT_TOOLSET_REGISTRY.resolve(
            "EMAIL_READING",
            runtime.tools,
        )
        self.assertTrue(resolution.is_available)

        with patch.object(
            runtime,
            "_append_draft",
            return_value={"status": "DRAFT_SAVED", "sent": False},
        ) as append_draft:
            result = await {
                tool.name: tool for tool in runtime.tools
            }["email_create_draft"].ainvoke(
                {
                    "to": ["recipient@example.com"],
                    "subject": "Test draft",
                    "body_text": "Saved, never sent.",
                }
            )
        self.assertEqual(result["status"], "DRAFT_SAVED")
        self.assertFalse(result["sent"])
        append_draft.assert_called_once()

        await runtime.stop()
        self.assertTrue(_FakeClient.contexts[0].exited)

    async def test_missing_upstream_read_tool_fails_closed(self) -> None:
        runtime = EmailMCPRuntime(_settings())
        with (
            patch("mcp_runtime.MultiServerMCPClient", _FakeClient),
            patch(
                "mcp_runtime.load_mcp_tools",
                AsyncMock(return_value=[]),
            ),
            patch.object(runtime, "_find_npx_command", return_value="npx.cmd"),
        ):
            with self.assertRaisesRegex(RuntimeError, "缺少必要工具"):
                await runtime.start()

        self.assertFalse(runtime.is_started)
        self.assertTrue(_FakeClient.contexts[0].exited)


if __name__ == "__main__":
    unittest.main()
