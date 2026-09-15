"""Opt-in public-page probes using the project's real Playwright MCP whitelist.

Isolated headless profile, cached npm package only, no models or account access.
Accept one JSON {tool, arguments} per stdin line; {stop: true} closes this session.
The operator must ground each navigation/control in an observed public page.
"""
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_runtime import PlaywrightMCPRuntime


class ProbeRuntime(PlaywrightMCPRuntime):
    def _build_server_config(self):
        config = super()._build_server_config()
        config["playwright"]["args"].append("--headless")
        return config


async def main():
    os.environ["npm_config_offline"] = "true"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    folder = Path(__file__).resolve().parents[1] / ".agent" / "browser-skill-probes" / (stamp + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    runtime = ProbeRuntime(profile_path=folder / "profile", output_path=folder / "output", owner_id=folder.name)
    index = 0
    try:
        await runtime.start()
        available = {tool.name: tool for tool in runtime.tools}
        schemas = {name: tool.args_schema.model_json_schema() if hasattr(tool.args_schema, "model_json_schema") else tool.args_schema for name, tool in available.items()}
        (folder / "schemas.json").write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"folder": str(folder), "schemas": schemas}, ensure_ascii=False), flush=True)
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            command = json.loads(line)
            if command.get("stop"):
                break
            name = command["tool"]
            if name not in available:
                raise ValueError("Tool is not in the runtime whitelist")
            index += 1
            try:
                result = await available[name].ainvoke(command.get("arguments", {}))
            except Exception as error:
                result = {"error": type(error).__name__, "detail": str(error)}
            record = {"at_utc": datetime.now(timezone.utc).isoformat(), "call": command, "result": result}
            path = folder / f"{index:03d}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(json.dumps({"evidence_file": str(path), "result": result}, ensure_ascii=False, default=str), flush=True)
    finally:
        await runtime.stop()
        print(json.dumps({"session_closed": True, "folder": str(folder)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
