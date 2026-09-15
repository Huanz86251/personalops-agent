"""Real project MCP download into isolated output; diagnose task-file visibility.

Only our local synthetic HTTP attachment, no login or private account access.
"""
import asyncio
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_browser_skill_paths import ProbeRuntime
from tools.local_native import _read


async def main():
    os.environ["npm_config_offline"] = "true"
    body = b"BROWSER_DOWNLOAD_BOUNDARY_ONLY\n"
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200)
            if self.path == "/attachment":
                data = body
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", 'attachment; filename="probe.txt"')
            else:
                data = b'<html><a href="/attachment">Download fixture</a></html>'
                self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    folder = Path(__file__).resolve().parents[1] / ".agent" / "browser-download-probes" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    runtime = ProbeRuntime(profile_path=folder / "profile", output_path=folder / "output", owner_id=folder.name)
    index = 0
    async def call(name, args):
        nonlocal index
        index += 1
        result = await available[name].ainvoke(args)
        (folder / f"{index:03d}.json").write_text(json.dumps({"tool": name, "arguments": args, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return result
    try:
        await runtime.start()
        available = {tool.name: tool for tool in runtime.tools}
        await call("browser_navigate", {"url": f"http://127.0.0.1:{server.server_port}/"})
        snapshot = await call("browser_snapshot", {})
        text = "\n".join(item.get("text", "") for item in snapshot if isinstance(item, dict))
        match = re.search(r'link "Download fixture" \[ref=([^\]]+)\]', text)
        assert match, text
        result = await call("browser_click", {"target": match.group(1), "element": "our synthetic attachment download"})
        await call("browser_snapshot", {})
        files = [path for path in (folder / "output").rglob("*") if path.is_file() and path.name == "probe.txt"]
        assert len(files) == 1, [str(p) for p in (folder / "output").rglob("*")]
        assert files[0].read_bytes() == body
        state = {"worker_id": "browser-" + folder.name, "files": {}}
        observations = {"browser_downloaded": True, "bytes_match": True, "browser_result_type": type(result).__name__,
                        "returns_state_command": hasattr(result, "update") and not isinstance(result, dict), "registered_downloads": len(state.get("worker_downloaded_artifacts", []))}
        for label, path in [("host", str(files[0])), ("unregistered_virtual", "/downloads/probe.txt")]:
            try:
                _read(path, SimpleNamespace(state=state))
                observations[label] = "UNEXPECTED_SUCCESS"
            except Exception as error:
                observations[label] = str(error)
        (folder / "summary.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"folder": str(folder), **observations}, ensure_ascii=False), flush=True)
    finally:
        await runtime.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        print(json.dumps({"browser_closed": True, "server_closed": not thread.is_alive()}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
