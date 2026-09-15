"""Real General graph reads one previously downloaded attachment, no mailbox/model calls.

The recorded MCP response is replayed; registration, parsing and filesystem are live.
Output is private evidence and a content-free summary. No sends or drafts.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from scripts.probe_file_routes import SequenceModel
from mcp_runtime import EMAIL_ATTACHMENT_PATH
from workers.general_runtime import GeneralStepRuntime
from eventing.store import AsyncEventStore


class BridgeModel(SequenceModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        for message in messages:
            if isinstance(message, ToolMessage) and message.tool_call_id == "route-0":
                self.actions[1]["args"]["file_path"] = json.loads(message.content)[
                    "private_files"
                ][0]["reading_path"]
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def main(record_path):
    evidence = json.loads(record_path.read_text("utf-8"))
    assert evidence["tool"] == "email_download_attachment"

    async def replay():
        return evidence["result"]

    tool = StructuredTool.from_function(
        coroutine=replay,
        name="email_download_attachment",
        description="Replay one authorized, already downloaded email attachment receipt",
        metadata={
            "task_file_source_root": str(EMAIL_ATTACHMENT_PATH.resolve()),
            "task_file_origin": "EMAIL",
        },
    )
    key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    root = Path(".agent/runtime-file-bridge-probes") / key
    root.mkdir(parents=True)
    store = AsyncEventStore(root / "events.sqlite3")
    await store.start()
    try:
        model = BridgeModel(
            actions=[
                {"tool": "email_download_attachment", "args": {}},
                {"tool": "read_file", "args": {"file_path": "pending"}},
            ]
        )
        runtime = GeneralStepRuntime(
            model,
            event_store=store,
            tools=[tool],
            progress_every_tool_calls=8,
            run_storage_root=root / "runs",
        )
        state = await runtime.ainvoke(
            {
                "event_id": key,
                "worker_id": "email-bridge-" + key,
                "step_id": "1",
                "messages": [
                    {
                        "role": "user",
                        "content": "Read the authorized attachment locally",
                    }
                ],
            }
        )
        results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        reading = json.loads(results[-1].content)
        assert reading.get("reading") and reading.get("preview"), results[-1].content
        assert all(isinstance(m.content, str) for m in results)
        summary = {
            "evidence": str(root.resolve()),
            "mailbox_calls": 0,
            "provider_calls": 0,
            "input": "previously downloaded real attachment receipt replay",
            "live_general_graph": True,
            "automatic_private_registration": True,
            "read_file_local_extraction": True,
            "reading": reading["reading"],
            "shared_publication": False,
            "raw_pdf_model_blocks": False,
        }
        (root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), "utf-8"
        )
        (root / "private-reading.json").write_text(
            json.dumps(reading, ensure_ascii=False, indent=2), "utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_record", type=Path)
    asyncio.run(main(parser.parse_args().download_record))
