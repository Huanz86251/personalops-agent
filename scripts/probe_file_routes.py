"""Model-free file-route diagnosis, isolated run and synthetic localhost server.

Real Web graph/tool HTTP download, reader, General runtime, publisher and optional
existing Docker pair. No account access, external sends or daemon/image setup.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_local_ocr import pdf_bytes
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from artifact_models import WorkerArtifactCandidate
from artifact_publisher import publish_artifact_to_handoff
from eventing.store import AsyncEventStore
from run_workspace import initialize_run_workspace
from tools.local_native import attachment_to_text, fetch_webpage, _read
from tools.web_tools import web_search_tool
from tools.web_artifacts import create_download_web_artifact_tool
from workers.submission import resolve_artifact_candidates
from workers.general_runtime import GeneralStepRuntime
from workers.web_runtime import WebStepRuntime


class SequenceModel(BaseChatModel):
    actions: list[dict]

    @property
    def _llm_type(self):
        return "scripted-file-route-probe"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        observed = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        for index, action in enumerate(self.actions):
            call_id = f"route-{index}"
            if call_id in observed:
                continue
            args = dict(action["args"])
            if args.get("path") == "$download":
                args["path"] = json.loads(observed["route-0"].content)["reading_path"]
            if args.get("file_path") == "$download":
                args["file_path"] = json.loads(observed["route-0"].content)["reading_path"]
            message = AIMessage(content="", tool_calls=[{"name": action["tool"], "args": args, "id": call_id}])
            break
        else:
            message = AIMessage(content="Scripted diagnostic finished; this is not an autonomous model assessment.")
        return ChatResult(generations=[ChatGeneration(message=message)])


class NoBrowserPool:
    @asynccontextmanager
    async def lease(self, worker_id):
        # This probe never calls browser tools; browser leasing is tested elsewhere.
        yield SimpleNamespace(tools=[])


async def main(with_docker):
    root = Path(__file__).resolve().parents[1]
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    folder = root / ".agent" / "file-route-probes" / name
    folder.mkdir(parents=True, exist_ok=False)
    findings = {}
    def save():
        (folder / "summary.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    def note(key, value):
        findings[key] = value
        save()
        print(json.dumps({key: value}, ensure_ascii=False, default=str), flush=True)
    fixture = pdf_bytes([None, None])
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path == "/redirect.pdf":
                self.send_response(302)
                self.send_header("Location", "/fixture.pdf")
                self.end_headers()
                return
            if self.path == "/large.pdf":
                content = fixture + b" " * (21 * 1024 * 1024)
            elif self.path == "/login.pdf":
                content = b"<html><body>Please sign in</body></html>"
            else:
                content = fixture
            self.send_response(200)
            self.send_header("Content-Type", "text/html" if self.path == "/login.pdf" else "application/pdf")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    store = AsyncEventStore(folder / "events.sqlite3")
    await store.start()
    try:
        note("evidence", str(folder))
        try:
            fetch_webpage.func(url + "/fixture.pdf")
        except Exception as error:
            note("fetch_loopback_rejected", str(error))
        web = WebStepRuntime(SequenceModel(actions=[
            {"tool": "download_web_artifact", "args": {"url": url + "/fixture.pdf"}},
            {"tool": "read_file", "args": {"file_path": "$download"}},
            {"tool": "attachment_to_text", "args": {"path": "$download", "output_path": "/artifacts/read.md", "max_pages": 1, "ocr": "off"}},
            {"tool": "read_file", "args": {"file_path": "/artifacts/read.md"}},
        ]), playwright_pool=NoBrowserPool(), event_store=store, tools=[web_search_tool, attachment_to_text],
            progress_every_tool_calls=8, run_storage_root=folder / "runs")
        state = await web.ainvoke({"event_id": name, "worker_id": "web-" + name, "step_id": "1", "messages": [{"role": "user", "content": "Run the bounded synthetic file diagnostic."}]})
        results = {m.tool_call_id: m.content for m in state["messages"] if isinstance(m, ToolMessage)}
        (folder / "web-tool-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        download = json.loads(results["route-0"])
        reading = json.loads(results["route-2"])
        assert download["status"] == "DOWNLOADED", download
        assert "Native heading" in results["route-3"], results
        note("web_graph", {"download": download["status"], "direct_read_file": results["route-1"],
                           "reader": reading["reading"], "converted_markdown_readable": True})
        original = state["worker_downloaded_artifacts"][0]
        other = SimpleNamespace(state={"worker_id": "other", "worker_downloaded_artifacts": [original]})
        try:
            _read(download["reading_path"], other)
        except Exception as error:
            note("other_worker_private_download_denied", str(error))
        layout = initialize_run_workspace(name, storage_root=folder / "runs")
        candidates = [WorkerArtifactCandidate.model_validate(download["artifact_candidate"]),
                      WorkerArtifactCandidate(candidate_id="parsed", kind="WORKSPACE_FILE", path="/artifacts/read.md", description="Parsed first page")]
        resolved = resolve_artifact_candidates(state, candidates)
        receipts = [publish_artifact_to_handoff(layout=layout, candidate=c) for c in resolved]
        note("manual_approval_real_publication", [r.model_dump(mode="json") for r in receipts])
        general = GeneralStepRuntime(SequenceModel(actions=[
            {"tool": "attachment_to_text", "args": {"path": receipts[0].handoff_path, "output_path": "/artifacts/page2.md", "start_page": 2, "max_pages": 1, "ocr": "off"}},
            {"tool": "read_file", "args": {"file_path": "/artifacts/page2.md"}},
            {"tool": "read_file", "args": {"file_path": receipts[1].handoff_path}},
        ]), event_store=store, tools=[attachment_to_text], progress_every_tool_calls=8, run_storage_root=folder / "runs")
        gstate = await general.ainvoke({"event_id": name, "worker_id": "general-" + name, "step_id": "2", "messages": [{"role": "user", "content": "Read the published handoff fixture."}]})
        gresults = {m.tool_call_id: m.content for m in gstate["messages"] if isinstance(m, ToolMessage)}
        (folder / "general-tool-results.json").write_text(json.dumps(gresults, ensure_ascii=False, indent=2), encoding="utf-8")
        assert "Native heading" in gresults["route-1"] and "Native heading" in gresults["route-2"]
        note("general_runtime", {"page2": json.loads(gresults["route-0"])["reading"], "published_markdown_readable": True})
        # Same actual download Tool and reader, bounded synthetic failures.
        tool = create_download_web_artifact_tool(max_file_mib=100)
        for case in ["login", "large", "redirect"]:
            current = {"worker_id": "edge-" + name}
            rt = SimpleNamespace(state=current, tool_call_id=case, stream_writer=lambda value: None)
            response = await tool.coroutine(url=url + f"/{case}.pdf", runtime=rt)
            current.update(response.update)
            value = json.loads(response.update["messages"][0].content)
            outcome = {"download_status": value["status"], "bytes": value.get("size_bytes")}
            try:
                parsed = attachment_to_text.func(path=value["reading_path"], output_path="/artifacts/check.md", runtime=rt, ocr="off")
                outcome["reader"] = json.loads(parsed.update["messages"][0].content)["reading"]
            except Exception as error:
                outcome["reader_error"] = str(error)
            note(case, outcome)
        if with_docker:
            from scripts.probe_code_skill_handoff import ExistingDaemonCLI
            from workers.docker_sandbox import CodeSandboxManager, CodeSandboxPolicy
            manager = CodeSandboxManager(CodeSandboxPolicy(auto_build=False, execute_timeout_seconds=30), cli=ExistingDaemonCLI())
            pair = None
            try:
                pair = manager.create_pair("route-" + name, handoff_root=layout.handoff_root)
                for role in ["WORKER", "REVIEWER"]:
                    if role == "REVIEWER":
                        pair = manager.handoff(pair, role)
                    backend = manager.worker_backend(pair) if role == "WORKER" else manager.reviewer_backend(pair)
                    files = backend.download_files([r.handoff_path for r in receipts])
                    assert all(f.error is None for f in files)
                    assert [hashlib.sha256(f.content).hexdigest() for f in files] == [r.sha256 for r in receipts]
                    denied = backend.upload_files([("/handoff/probe-write.txt", b"denied")])
                    assert all(f.error is not None for f in denied)
                    env = backend.execute("python -c 'import importlib.util; print({n:importlib.util.find_spec(n) is not None for n in [\"pypdf\",\"fitz\",\"rapidocr_onnxruntime\"]})'", timeout=30)
                    note("docker_" + role.lower(), {"published_bytes_match": True, "handoff_write_denied": True, "pdf_dependencies": env.output})
            finally:
                if pair:
                    manager.cleanup(pair)
                    owned = [("container", pair.worker_container), ("container", pair.reviewer_container), ("volume", pair.candidate_volume), ("volume", pair.review_volume)]
                    note("docker_cleanup", {resource: manager._inspect_owned_resource(kind, resource) is None for kind, resource in owned})
        note("finished", True)
    finally:
        await store.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        note("local_server_closed", not thread.is_alive())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.docker))
