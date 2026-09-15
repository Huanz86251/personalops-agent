"""Live project browser MCP + Web graph automatic private download/read test.

Uses only an isolated local synthetic website; no account or model calls.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import ToolMessage
from scripts.probe_browser_skill_paths import ProbeRuntime
from scripts.probe_file_routes import SequenceModel
from workers.web_runtime import WebStepRuntime
from eventing.store import AsyncEventStore
from tools.web_tools import web_search_tool


class BrowserModel(SequenceModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            text = (
                message.content
                if isinstance(message.content, str)
                else "\n".join(b.get("text", "") for b in message.content)
            )
            match = re.search(r'link "Download fixture" \[ref=([^\]]+)\]', text)
            if match:
                self.actions[2]["args"]["target"] = match.group(1)
            index = text.find('{"private_files":')
            if index >= 0:
                self.actions[4]["args"]["file_path"] = json.loads(text[index:])[
                    "private_files"
                ][0]["reading_path"]
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def main():
    os.environ["npm_config_offline"] = "true"
    body = b"BROWSER_FILE_BRIDGE_SUCCESS\n"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            if self.path == "/attachment":
                data = body
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header(
                    "Content-Disposition", 'attachment; filename="probe.txt"'
                )
            else:
                data = b'<html><a href="/attachment">Download fixture</a></html>'
                self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    root = Path(".agent/browser-file-bridge-probes").resolve() / key
    root.mkdir(parents=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    browser = ProbeRuntime(
        profile_path=root / "profile", output_path=root / "output", owner_id=key
    )

    class Pool:
        @asynccontextmanager
        async def lease(self, worker_id):
            yield browser

    store = AsyncEventStore(root / "events.sqlite3")
    await store.start()
    try:
        await browser.start()
        model = BrowserModel(
            actions=[
                {
                    "tool": "browser_navigate",
                    "args": {"url": f"http://127.0.0.1:{server.server_port}/"},
                },
                {"tool": "browser_snapshot", "args": {}},
                {
                    "tool": "browser_click",
                    "args": {
                        "target": "pending",
                        "element": "synthetic fixture download",
                    },
                },
                {"tool": "browser_snapshot", "args": {}},
                {"tool": "read_file", "args": {"file_path": "pending"}},
            ]
        )
        runtime = WebStepRuntime(
            model,
            playwright_pool=Pool(),
            event_store=store,
            tools=[web_search_tool],
            progress_every_tool_calls=8,
            run_storage_root=root / "runs",
        )
        state = await runtime.ainvoke(
            {
                "event_id": key,
                "worker_id": "browser-" + key,
                "step_id": "1",
                "messages": [
                    {"role": "user", "content": "Download and read our fixture"}
                ],
            }
        )
        results = {
            m.tool_call_id: m.content
            for m in state["messages"]
            if isinstance(m, ToolMessage)
        }
        (root / "tool-results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), "utf-8"
        )
        assert "BROWSER_FILE_BRIDGE_SUCCESS" in str(results["route-4"]), results
        summary = {
            "evidence": str(root),
            "real_project_browser_mcp": True,
            "real_web_graph": True,
            "automatic_private_registration": len(state["worker_downloaded_artifacts"])
            == 1,
            "read_file_contains_downloaded_text": True,
            "shared_publication": False,
            "provider_calls": 0,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
        print(json.dumps(summary), flush=True)
    finally:
        await browser.stop()
        await store.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        print(
            json.dumps(
                {"browser_closed": True, "server_closed": not thread.is_alive()}
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
