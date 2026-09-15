"""Live public PDF download + read_file in the project's Web graph, no model API."""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_file_routes import SequenceModel, NoBrowserPool
from workers.web_runtime import WebStepRuntime
from tools.web_tools import web_search_tool
from eventing.store import AsyncEventStore
from langchain_core.messages import ToolMessage


async def main():
    key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    root = Path(".agent/public-pdf-read-probes").resolve() / key
    root.mkdir(parents=True)
    async with AsyncEventStore(root / "events.sqlite3") as store:
        model = SequenceModel(
            actions=[
                {
                    "tool": "download_web_artifact",
                    "args": {
                        "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
                    },
                },
                {"tool": "read_file", "args": {"file_path": "$download"}},
            ]
        )
        runtime = WebStepRuntime(
            model,
            event_store=store,
            playwright_pool=NoBrowserPool(),
            tools=[web_search_tool],
            progress_every_tool_calls=8,
            run_storage_root=root / "runs",
        )
        state = await runtime.ainvoke(
            {
                "event_id": key,
                "worker_id": "web-" + key,
                "step_id": "1",
                "messages": [
                    {"role": "user", "content": "Read the public W3C PDF test fixture"}
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
        assert "Dummy PDF file" in results["route-1"], results
        download = json.loads(results["route-0"])
        summary = {
            "evidence": str(root),
            "live_http_download": True,
            "real_web_graph": True,
            "local_read_file": True,
            "sha256": download["sha256"],
            "bytes": download["size_bytes"],
            "reading_path": download["reading_path"],
            "provider_calls": 0,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
