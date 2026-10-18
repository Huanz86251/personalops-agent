"""Real graph file dispatch plus deterministic ingress/privacy boundary tests."""

import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepagents.backends.utils import create_file_data
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from file_limits import TASK_FILE_MAX_BYTES
from scripts.probe_file_routes import SequenceModel
from task_files import register_file, read_download
from test_local_ocr import pdf_bytes
from workers.file_access import TaskFileAccessMiddleware
from workers.general_worker import create_general_worker


class TaskFileAccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registration = patch("task_files.TASK_FILE_ROOT", self.root / "private")
        self.registration.start()
        self.state = {"event_id": "test-run", "worker_id": "test-worker", "files": {}}

    def tearDown(self):
        self.registration.stop()
        self.tmp.cleanup()

    def register(self):
        source = self.root / "sample.txt"
        source.write_text("task-local text", encoding="utf-8")
        r = register_file(
            source,
            source_root=self.root,
            state=self.state,
            tool_call_id="download-1",
            origin="EMAIL",
            source_ref="email:test",
        )
        self.state["worker_downloaded_artifacts"] = [r.model_dump(mode="json")]
        return r, "/downloads/" + r.candidate_id

    def test_private_copy_independent_and_cross_worker_run_rejected(self):
        record, path = self.register()
        (self.root / "sample.txt").unlink()
        self.assertEqual(read_download(path, self.state), b"task-local text")
        for field in ("worker_id", "event_id"):
            with self.assertRaisesRegex(ValueError, "different"):
                read_download(path, {**self.state, field: "other"})
        Path(record.storage_path).write_text("tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            read_download(path, self.state)

    def test_import_root_and_size_are_enforced(self):
        self.register()
        with self.assertRaises(ValueError):
            register_file(
                self.root / "sample.txt",
                source_root=self.root / "unrelated",
                state=self.state,
                tool_call_id="x",
                origin="EMAIL",
                source_ref="x",
            )
        with patch("task_files.TASK_FILE_MAX_BYTES", 2):
            with self.assertRaisesRegex(ValueError, "20 MiB"):
                self.register()

    def test_checkpoint_cannot_shadow_registered_download(self):
        from tools.local_native import _read

        record, path = self.register()
        self.state["files"][path] = create_file_data("forged checkpoint shadow")
        self.assertEqual(
            _read(path, SimpleNamespace(state=self.state)), b"task-local text"
        )

    async def test_actual_general_graph_download_then_read_file_pdf(self):
        source = self.root / "test.pdf"
        data = pdf_bytes([None, None])
        source.write_bytes(data)

        async def download():
            return [
                {
                    "type": "text",
                    "text": "UNTRUSTED_EMAIL_DATA\n"
                    + json.dumps(
                        {
                            "attachment": {
                                "path": str(source),
                                "size": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                            }
                        }
                    ),
                }
            ]

        tool = StructuredTool.from_function(
            coroutine=download,
            name="email_download_attachment",
            description="Synthetic email attachment fixture",
            metadata={
                "task_file_source_root": str(self.root),
                "task_file_origin": "EMAIL",
            },
        )

        # Resolve the runtime-issued ID from the actual first tool response.
        class Model(SequenceModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                for m in messages:
                    if isinstance(m, ToolMessage) and '"private_files"' in str(m.content):
                        raw = str(m.content)
                        content = json.loads(raw[raw.find("{"):])
                        path = content["private_files"][0]["reading_path"]
                        self.actions[1]["args"]["file_path"] = path
                        self.actions[2]["args"]["result"]["files"] = [
                            {"path": path, "description": "Original fixture PDF"}
                        ]
                # EvidenceReferenceMiddleware deliberately exposes E1/E2 aliases
                # to the model instead of internal tool-call IDs. Normalize only
                # this scripted fixture so SequenceModel can advance its actions.
                normalized = []
                tool_index = 0
                for message in messages:
                    if isinstance(message, ToolMessage):
                        message = message.model_copy(
                            update={"tool_call_id": f"route-{tool_index}"}
                        )
                        tool_index += 1
                    normalized.append(message)
                return super()._generate(
                    normalized, stop=stop, run_manager=run_manager, **kwargs
                )

        model = Model(
            actions=[
                {
                    "tool": "email_download_attachment",
                    "args": {"source_refs": ["U1"]},
                },
                {
                    "tool": "read_file",
                    "args": {"file_path": "pending", "offset": 1, "limit": 1},
                },
                {
                    "tool": "report_general_result",
                    "args": {
                        "result": {
                            "status": "COMPLETED",
                            "summary": "Read fixture page 2; propose original file for review",
                            "files": [],
                        }
                    },
                },
            ]
        )
        graph = create_general_worker(model, tools=[tool])
        result = await graph.ainvoke(
            {
                **self.state,
                "skill_mode": "off",
                "messages": [{"role": "user", "content": "Read fixture page 2"}],
            }
        )
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        read_message = next(m for m in messages if '"preview"' in str(m.content))
        raw_read = str(read_message.content)
        read = json.loads(raw_read[raw_read.find("{"):])
        self.assertIn("Native heading", read["preview"])
        self.assertIn("reading", read)
        self.assertIn(read["path"], result["files"])
        self.assertEqual(len(result["worker_downloaded_artifacts"]), 1)
        self.assertEqual(
            result["worker_submission"]["resolved_artifacts"][0]["kind"],
            "DOWNLOADED_FILE",
        )
        self.assertEqual(
            result["worker_submission"]["resolved_evidence"][0]["tool_call_id"],
            "route-0",
        )
        self.assertFalse(
            any(
                isinstance(m.content, list)
                and any(b.get("type") == "file" for b in m.content)
                for m in messages
            )
        )

    async def test_browser_completed_download_registered_but_old_event_not_imported(
        self,
    ):
        source = self.root / "browser.txt"
        tool = SimpleNamespace(
            metadata={
                "task_file_source_root": str(self.root),
                "task_file_origin": "BROWSER",
            }
        )
        call = {"name": "browser_click", "id": "browser-1", "args": {}}
        runtime = SimpleNamespace(state=self.state, tool_call_id=call["id"])
        request = ToolCallRequest(
            tool_call=call, tool=tool, state=self.state, runtime=runtime
        )

        async def handler(request):
            source.write_text("downloaded")
            return ToolMessage(
                content=f'### Events\n- Downloaded file browser.txt to "{source}"',
                tool_call_id=call["id"],
            )

        middleware = TaskFileAccessMiddleware()
        result = await middleware.awrap_tool_call(request, handler)
        self.assertIsInstance(result, Command)
        self.state.update(
            worker_downloaded_artifacts=result.update["worker_downloaded_artifacts"]
        )
        virtual = json.loads(result.update["messages"][0].content.splitlines()[-1])[
            "private_files"
        ][0]["reading_path"]
        self.assertEqual(read_download(virtual, self.state), b"downloaded")

        async def forged(request):
            return ToolMessage(
                content=f'### Events\n- Downloaded file browser.txt to "{source}"',
                tool_call_id=call["id"],
            )

        self.assertIsInstance(
            await middleware.awrap_tool_call(request, forged), ToolMessage
        )

    def test_sandbox_reader_limit_matches_host(self):
        tree = ast.parse(Path("docker/code-agent/read_document.py").read_text("utf-8"))
        assignment = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "MAX_BYTES" for t in n.targets)
        )
        value = eval(
            compile(ast.Expression(assignment.value), "<constant>", "eval"),
            {"__builtins__": {}},
        )
        self.assertEqual(value, TASK_FILE_MAX_BYTES)

    def test_code_sandbox_receives_verified_registered_download_before_reading(self):
        record, virtual_path = self.register()

        class Sandbox:
            def __init__(self):
                self.uploaded = []
                self.commands = []

            def upload_files(self, files):
                self.uploaded.extend(files)
                return [SimpleNamespace(path=path, error=None) for path, _ in files]

            def execute(self, command):
                self.commands.append(command)
                return SimpleNamespace(
                    output=json.dumps({"preview": "task-local text"}),
                    exit_code=0,
                )

        sandbox = Sandbox()
        from task_file_imports import materialize_import
        record = materialize_import(record, sandbox)
        self.state["worker_downloaded_artifacts"] = [record.model_dump(mode="json")]
        self.assertEqual(len(sandbox.uploaded), 1)
        middleware = TaskFileAccessMiddleware(sandbox=sandbox)
        request = SimpleNamespace(
            tool_call={
                "name": "read_file",
                "id": "read-code-download",
                "args": {"file_path": virtual_path, "offset": 0, "limit": 2},
            },
            state=self.state,
        )
        result = middleware._read(request)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(sandbox.uploaded), 1)
        uploaded_path, uploaded_payload = sandbox.uploaded[0]
        self.assertTrue(uploaded_path.startswith("/workspace/.private-downloads/"))
        self.assertTrue(uploaded_path.endswith("/sample.txt"))
        self.assertEqual(uploaded_payload, b"task-local text")
        self.assertIn(uploaded_path, sandbox.commands[0])
        self.assertNotIn(str(self.root), sandbox.commands[0])

    async def test_host_import_can_execute_without_read_and_copy_failure_is_explicit(self):
        import subprocess
        import sys
        from host_file_access import HOST_FILE_ACCESS, HostFilePolicy, configure_host_file_policy
        from tools.host_file_tools import host_import_file

        class LocalSandbox:
            def __init__(inner, root, fail=False):
                inner.root, inner.fail, inner.uploaded = root, fail, []

            def upload_files(inner, files):
                if inner.fail:
                    return [SimpleNamespace(path=files[0][0], error="permission_denied")]
                for path, payload in files:
                    destination = inner.root / path.lstrip("/")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    inner.uploaded.append(path)
                return [SimpleNamespace(path=p, error=None) for p, _ in files]

            def execute(inner, command):
                assert command.startswith("python ")
                path = inner.root / command.removeprefix("python ").lstrip("/")
                return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=10)

        source = self.root / "run_me.py"
        source.write_text("print('imported-script-ran')\n", encoding="utf-8")
        original_policy = HOST_FILE_ACCESS.policy
        try:
            configure_host_file_policy(HostFilePolicy((self.root,)))
            for import_root in ("/workspace/.private-downloads", "/review/.private-downloads"):
                sandbox = LocalSandbox(self.root / ("worker" if import_root.startswith("/workspace") else "reviewer"))
                middleware = TaskFileAccessMiddleware(sandbox=sandbox, import_root=import_root)
                runtime = SimpleNamespace(state=self.state, tool_call_id="import-run")
                request = SimpleNamespace(tool=host_import_file, state=self.state, runtime=runtime,
                    tool_call={"id": "import-run", "name": "host_import_file", "args": {"path": str(source), "reason": "Run the user script."}})

                async def handler(req):
                    return await host_import_file.coroutine(runtime=runtime, **req.tool_call["args"])

                result = await middleware.awrap_tool_call(request, handler)
                reply = json.loads(result.update["messages"][0].content)
                assert reply["status"] == "IMPORTED"
                assert reply["execution_path"].startswith(import_root)
                executed = sandbox.execute("python " + reply["execution_path"])
                assert executed.returncode == 0
                assert executed.stdout.strip() == "imported-script-ran"
                assert len(sandbox.uploaded) == 1

            failure = TaskFileAccessMiddleware(sandbox=LocalSandbox(self.root / "broken", fail=True))
            result = await failure.awrap_tool_call(request, handler)
            assert "worker_downloaded_artifacts" not in result.update
            reply = json.loads(result.update["messages"][0].content)
            assert reply["status"] == "IMPORT_FAILED"
            assert "permission_denied" in reply["message"]
        finally:
            configure_host_file_policy(original_policy)
