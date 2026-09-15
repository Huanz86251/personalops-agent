"""Read-only public Web probes through the project's actual registered tools.

No model, MCP session, login, backend override, or business write. Persist each
sample separately under Git-ignored .agent; never overwrite earlier evidence.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.web_tools import web_search_tool
from tools.local_native import fetch_webpage


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["search", "fetch"])
    parser.add_argument("targets", nargs="+", help="Public queries or URLs; do not supply private data")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if len(args.targets) > 12:
        parser.error("At most 12 sequential probes per invocation")
    folder = Path(__file__).resolve().parents[1] / ".agent" / "web-source-probes"
    folder.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        started = time.monotonic()
        inputs = ({"query": target, "page": 1} if args.kind == "search"
                  else {"url": target, "start_index": args.start_index, "max_length": 12000})
        selected_tool = web_search_tool if args.kind == "search" else fetch_webpage
        try:
            result = await selected_tool.ainvoke(inputs)
        except Exception as error:
            result = {"probe_exception": type(error).__name__, "detail": str(error)}
        stamp = datetime.now(timezone.utc)
        evidence = {"at_utc": stamp.isoformat(), "tool": selected_tool.name,
                    "input": inputs, "elapsed_seconds": round(time.monotonic() - started, 3),
                    "result": result}
        serialized = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
        path = folder / f"{stamp:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        summary = {k: v for k, v in evidence.items() if k != "result"}
        summary["evidence_file"] = str(path)
        summary["sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
        if args.kind == "search" or not isinstance(result, dict) or "content" not in result:
            summary["result"] = result
        else:
            summary["result"] = {k: v for k, v in result.items() if k != "content"}
            body = result["content"]
            summary["content_head"] = body[:1200]
            summary["content_tail"] = body[-800:]
        print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
