"""Provider-free checks of real local conversions, tool wiring and file boundaries."""

import base64
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from deepagents.backends.utils import create_file_data
from langchain_core.tools import ToolException

from tools import ALL_TOOLS
from tools import local_native as local


class NativeToolTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(state={"files": {}}, tool_call_id="local-test")

    def apply(self, command):
        self.runtime.state["files"].update(command.update["files"])
        return json.loads(command.update["messages"][0].content)

    def put(self, path, content):
        self.runtime.state["files"][path] = create_file_data(content)

    def test_schema_hides_runtime_and_registry_is_unique(self):
        names = [t.name for t in ALL_TOOLS]
        self.assertEqual(len(names), len(set(names)))
        for current in local.LOCAL_NATIVE_TOOLS:
            self.assertIn(current.name, names)
            self.assertNotIn("runtime", current.args)

    def test_native_tools_not_injected_into_explicit_evaluation_toolset(self):
        from workers.general_worker import select_general_worker_tools

        # Explicit input stays explicit; no native-tool injection in Worker builder.
        self.assertEqual(select_general_worker_tools([]), [])
        from workers.web_worker import select_web_worker_tools

        selected = {t.name for t in select_web_worker_tools(ALL_TOOLS)}
        self.assertIn("fetch_webpage", selected)
        self.assertNotIn("spreadsheet_write", selected)

    def test_text_attachment_and_saved_preview(self):
        self.put("/input.txt", "Local attachment: 你好\n" + "abc " * 4000)
        result = self.apply(
            local.attachment_to_text.func(
                "/input.txt", "/artifacts/read.md", self.runtime
            )
        )
        self.assertTrue(result["truncated"])
        self.assertGreater(
            len(local._read("/artifacts/read.md", self.runtime)), local.MAX_TEXT
        )

    def test_markdown_docx_roundtrip(self):
        self.put(
            "/input.md",
            "# 本地报告\n\n这是本地转换。\n\n| Item | Count |\n|---|---|\n| A | 3 |",
        )
        self.apply(
            local.convert_document.func(
                "/input.md", "/artifacts/report.docx", self.runtime
            )
        )
        self.assertTrue(
            local._read("/artifacts/report.docx", self.runtime).startswith(b"PK")
        )
        result = self.apply(
            local.attachment_to_text.func(
                "/artifacts/report.docx", "/artifacts/report.md", self.runtime
            )
        )
        self.assertIn("本地报告", result["preview"])
        self.assertIn("3", result["preview"])

    def test_spreadsheet_create_format_chart_read_and_extract(self):
        self.apply(
            local.spreadsheet_write.func(
                "/artifacts/table.xlsx",
                "统计",
                [["项目", "数量"], ["A", 3], ["B", 5]],
                self.runtime,
            )
        )
        data = local.spreadsheet_read.func(
            "/artifacts/table.xlsx", self.runtime, sheet="统计", max_rows=2
        )
        self.assertEqual(data["next_row"], 3)
        self.assertEqual(data["rows"][1], ["A", 3])
        self.apply(
            local.spreadsheet_format.func(
                "/artifacts/table.xlsx",
                "/artifacts/styled.xlsx",
                "统计",
                "A1:B1",
                self.runtime,
                bold=True,
            )
        )
        self.apply(
            local.spreadsheet_chart.func(
                "/artifacts/styled.xlsx",
                "/artifacts/chart.xlsx",
                "统计",
                "A1:B3",
                self.runtime,
            )
        )
        wb = local._workbook("/artifacts/chart.xlsx", self.runtime)
        self.assertTrue(wb["统计"]["A1"].font.bold)
        self.assertEqual(len(wb["统计"]._charts), 1)
        wb.close()
        result = self.apply(
            local.attachment_to_text.func(
                "/artifacts/chart.xlsx", "/artifacts/table.md", self.runtime
            )
        )
        self.assertIn("数量", result["preview"])

    def test_existing_output_requires_explicit_overwrite(self):
        self.apply(
            local.spreadsheet_write.func(
                "/artifacts/table.xlsx", "Sheet1", [[1]], self.runtime
            )
        )
        before = local._read("/artifacts/table.xlsx", self.runtime)
        with self.assertRaises(ToolException):
            local.spreadsheet_write.func(
                "/artifacts/table.xlsx", "Sheet1", [[2]], self.runtime
            )
        self.assertEqual(before, local._read("/artifacts/table.xlsx", self.runtime))
        self.apply(
            local.spreadsheet_write.func(
                "/artifacts/table.xlsx",
                "Sheet1",
                [[2]],
                self.runtime,
                source_path="/artifacts/table.xlsx",
                overwrite=True,
            )
        )
        self.assertEqual(
            local.spreadsheet_read.func(
                "/artifacts/table.xlsx", self.runtime, sheet="Sheet1"
            )["rows"],
            [[2]],
        )

    def test_host_paths_traversal_and_shared_output_rejected(self):
        for value in [
            "C:/private.txt",
            "../private.txt",
            "/handoff/../../private",
            "\\\\host\\secret",
        ]:
            with self.subTest(path=value), self.assertRaises(ValueError):
                local._read(value, self.runtime)
        with self.assertRaises(ValueError):
            local._save("/handoff/test.txt", b"bad", self.runtime)

    def test_run_handoff_is_read_only_and_scoped(self):
        from run_workspace import initialize_run_workspace

        with TemporaryDirectory() as directory:
            layout = initialize_run_workspace(
                "run-a",
                storage_root=Path(directory),
                canonical_root=Path(directory) / "canonical",
            )
            (layout.handoff_root / "input.txt").write_text("shared", encoding="utf-8")
            self.runtime.state.update(event_id="run-a", run_storage_root=directory)
            self.assertEqual(local._read("/handoff/input.txt", self.runtime), b"shared")
            self.runtime.state["event_id"] = "run-b"
            with self.assertRaises(FileNotFoundError):
                local._read("/handoff/input.txt", self.runtime)

    def test_math_is_symbolic_and_rejects_code(self):
        self.assertEqual(
            local.symbolic_math.func("x**2 - 1", "solve")["result"], "[-1, 1]"
        )
        self.assertEqual(
            local.symbolic_math.func("sin(x)", "differentiate")["result"], "cos(x)"
        )
        for text in [
            "__import__('os').getcwd()",
            "x.__class__",
            "[x for x in y]",
            "2**1000",
            "((10000**12)**12)",
        ]:
            with self.subTest(text=text), self.assertRaises(ToolException):
                local.symbolic_math.func(text)

    def test_syntax_check_never_executes(self):
        with patch("builtins.print") as printer:
            result = local.python_syntax_check.func('print("never executed")')
        printer.assert_not_called()
        self.assertTrue(result["valid_syntax"])
        self.assertFalse(local.python_syntax_check.func("if :")["valid_syntax"])
        self.assertFalse(local.python_syntax_check.func("return 1")["valid_syntax"])

    def test_graph_saves_native_binary_without_model_echo(self):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        from workers.general_worker import create_general_worker

        class WorkbookModel(BaseChatModel):
            @property
            def _llm_type(self):
                return "scripted-local-workbook"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                if any(isinstance(m, ToolMessage) for m in messages):
                    message = AIMessage(content="Workbook generated for review.")
                else:
                    message = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "spreadsheet_write",
                                "id": "create-xlsx",
                                "type": "tool_call",
                                "args": {
                                    "output_path": "/artifacts/demo.xlsx",
                                    "sheet": "Demo",
                                    "rows": [["A", 2]],
                                },
                            }
                        ],
                    )
                return ChatResult(generations=[ChatGeneration(message=message)])

        graph = create_general_worker(WorkbookModel(), tools=[local.spreadsheet_write])
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Make the local spreadsheet."}], "skill_mode": "off"}
        )
        data = result["files"]["/artifacts/demo.xlsx"]
        self.assertEqual(data["encoding"], "base64")
        self.assertTrue(base64.b64decode(data["content"]).startswith(b"PK"))
        observation = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertLess(len(observation.content), 1000)

    def test_ruff_finds_real_undefined_variable(self):
        result = local.python_static_check.func("print(missing_name)\n")
        self.assertIn("F821", [row["code"] for row in result["diagnostics"]])
        self.assertFalse(result["executed"])

    def test_fetch_chunks_strips_scripts_and_checks_redirects(self):
        import httpx

        factory = httpx.Client
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<h1>Title</h1><script>secret_script()</script><p>body</p>",
            )
        )
        with (
            patch.object(local, "_public_url"),
            patch(
                "httpx.Client",
                side_effect=lambda **kwargs: factory(transport=transport, **kwargs),
            ),
        ):
            result = local.fetch_webpage.func("https://example.com", max_length=8)
            self.assertEqual(result["next_index"], 8)
            self.assertNotIn("secret_script", result["content"])
        with self.assertRaises(ToolException):
            local.fetch_webpage.func("http://127.0.0.1")

    def test_chinese_html_meta_charset_is_preserved(self):
        import httpx
        factory = httpx.Client
        html = '<html><meta charset="gb2312"><p>开放时间与地铁路线</p></html>'.encode("gb2312")
        transport = httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "text/html"}, content=html))
        with patch.object(local, "_public_url"), patch("httpx.Client", side_effect=lambda **kw: factory(transport=transport, **kw)):
            result = local.fetch_webpage.func("https://example.com/visit")
        self.assertIn("开放时间与地铁路线", result["content"])

    def test_generated_workbook_resolves_as_existing_artifact_kind(self):
        from artifact_models import WorkerArtifactCandidate
        from workers.submission import resolve_artifact_candidates

        with TemporaryDirectory() as directory:
            self.runtime.state.update(
                event_id="test-run", worker_id="worker-a", run_storage_root=directory
            )
            self.apply(
                local.spreadsheet_write.func(
                    "/artifacts/table.xlsx", "Sheet1", [[1]], self.runtime
                )
            )
            records = resolve_artifact_candidates(
                self.runtime.state,
                [
                    WorkerArtifactCandidate(
                        candidate_id="table",
                        kind="WORKSPACE_FILE",
                        description="Local XLSX",
                        path="/artifacts/table.xlsx",
                    )
                ],
            )
            self.assertTrue(
                Path(records[0].storage_path).read_bytes().startswith(b"PK")
            )


if __name__ == "__main__":
    unittest.main()
