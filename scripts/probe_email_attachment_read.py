"""Explicitly authorized, bounded live read/download probe. Never sends or drafts.

Checks at most ten recent message attachment lists; downloads one supported file
<= 2 MiB. Private results stay in .agent. No message bodies or snippets fetched.
The separate diagnostic import is NOT an existing production email bridge.
"""
import asyncio
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings
from mcp_runtime import EmailMCPRuntime, EMAIL_ATTACHMENT_PATH


def payload(result):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict) and "content" in result:
        result = result["content"]
    if isinstance(result, list):
        result = next(item["text"] for item in result if item.get("type") == "text")
    if isinstance(result, str):
        if result.startswith("UNTRUSTED_EMAIL_DATA:"):
            result = result.split("\n", 1)[1]
        return json.loads(result)
    return result


async def main():
    settings = load_settings().email_mcp
    print(json.dumps({"enabled": settings.enabled, "configured": settings.is_configured}), flush=True)
    if not settings.enabled or not settings.is_configured:
        return
    root = Path(__file__).resolve().parents[1] / ".agent" / "email-read-probes"
    folder = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    runtime = EmailMCPRuntime(settings)
    index = 0
    async def call(name, args):
        nonlocal index
        assert name in {"email_connection_status", "email_list_recent", "email_list_attachments", "email_download_attachment"}
        index += 1
        result = await asyncio.wait_for(available[name].ainvoke(args), timeout=60)
        (folder / f"{index:03d}.json").write_text(json.dumps({"tool": name, "arguments": args, "result": result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"tool": name, "evidence_file": str(folder / f"{index:03d}.json")}), flush=True)
        return payload(result)
    try:
        await asyncio.wait_for(runtime.start(), timeout=60)
        available = {tool.name: tool for tool in runtime.tools}
        schemas = {name: tool.args_schema.model_json_schema() if hasattr(tool.args_schema, "model_json_schema") else tool.args_schema for name, tool in available.items()}
        (folder / "schemas.json").write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
        await call("email_connection_status", {})
        messages = await call("email_list_recent", {"limit": 10, "includeSnippet": False})
        chosen = None
        for message in messages:
            listing = await call("email_list_attachments", {"id": message["id"]})
            for item in listing["attachments"]:
                suffix = Path(item.get("filename") or "").suffix.lower()
                if suffix in {".txt", ".pdf", ".docx", ".xlsx", ".csv"} and 0 < item.get("size", 0) <= 2 * 1024 * 1024:
                    chosen = (message["id"], item)
                    break
            if chosen:
                break
        if not chosen:
            print(json.dumps({"outcome": "No supported attachment <=2MiB in bounded recent window", "folder": str(folder)}), flush=True)
            return
        uid, item = chosen
        downloaded = await call("email_download_attachment", {"id": uid, "part": item["part"], "maxBytes": 2 * 1024 * 1024})
        attachment = downloaded["attachment"]
        actual = Path(attachment["path"]).resolve()
        actual.relative_to(EMAIL_ATTACHMENT_PATH.resolve())
        data = actual.read_bytes()
        assert len(data) == attachment["size"]
        assert hashlib.sha256(data).hexdigest() == attachment["sha256"]
        from tools.local_native import attachment_to_text
        state = {"files": {}, "worker_id": "email-probe"}
        tool_runtime = SimpleNamespace(state=state, tool_call_id="email-read-probe")
        result = {"downloaded": True, "bytes": len(data), "sha256_verified": True, "suffix": actual.suffix, "model_calls": 0}
        try:
            attachment_to_text.func(path=str(actual), output_path="/artifacts/probe.md", runtime=tool_runtime, ocr="off")
            result["production_host_path_read"] = "UNEXPECTED_SUCCESS"
        except Exception as error:
            result["production_host_path_read"] = str(error)
        # Controlled diagnostic ONLY: manually import bytes in an isolated state.
        from deepagents.backends.utils import create_file_data
        key = "/inputs/attachment" + actual.suffix.lower()
        record = create_file_data(base64.b64encode(data).decode("ascii"))
        record["encoding"] = "base64"
        state["files"][key] = record
        converted = attachment_to_text.func(path=key, output_path="/artifacts/probe.md", runtime=tool_runtime, ocr="off")
        (folder / "diagnostic-conversion.json").write_text(json.dumps(converted.update, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        result["diagnostic_import_read"] = "converted after MANUAL import; not a production bridge"
        result["generated_files"] = list(converted.update.get("files", {}))
        (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"folder": str(folder), **result}, ensure_ascii=False), flush=True)
    finally:
        await runtime.stop()
        print(json.dumps({"email_session_closed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live-download", action="store_true", required=True,
                        help="Use only after explicit authorization to read recent metadata and download one attachment.")
    parser.parse_args()
    asyncio.run(main())
