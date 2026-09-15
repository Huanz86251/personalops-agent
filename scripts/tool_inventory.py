"""Print declared tools and role exposure without starting MCP or calling models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.utils.function_calling import convert_to_openai_tool

from mcp_runtime import PLAYWRIGHT_ALLOWED_TOOLS
from tools import ALL_TOOLS
from workers.code_worker import CODE_WORKER_FILESYSTEM_TOOLS
from workers.general_worker import DEEP_AGENT_FILE_TOOLS, LEGACY_TOOL_NAMES
from workers.web_worker import WEB_BUSINESS_TOOL_NAMES
from toolsets import DEFAULT_TOOLSET_REGISTRY


def inventory():
    records = []
    for current in ALL_TOOLS:
        name = current.name
        roles = []
        if name not in LEGACY_TOOL_NAMES:
            roles.append("GENERAL")
        if name in WEB_BUSINESS_TOOL_NAMES:
            roles.append("WEB")
        records.append(
            {
                "name": name,
                "roles": roles,
                "condition": "owner/chat/roots configured + authenticated event; upload requires separate confirmation"
                if name == "send_local_file_to_feishu"
                else "registered; role filter applies",
                "legacy_only": name in LEGACY_TOOL_NAMES and not roles,
                "metadata": current.metadata or {},
                "schema": convert_to_openai_tool(current),
            }
        )
    toolsets = []
    registered_names = {item["name"] for item in records}
    for name in DEFAULT_TOOLSET_REGISTRY.toolset_names:
        spec = DEFAULT_TOOLSET_REGISTRY.get(name)
        toolsets.append(
            {
                "name": spec.name,
                "description": spec.description,
                "routing_threshold": spec.routing_threshold,
                "required_tool_names": list(spec.required_tool_names),
                "optional_tool_names": list(spec.optional_tool_names),
                "available_from_registered_tools": set(spec.required_tool_names).issubset(
                    registered_names
                ),
                "schema_lookup": (
                    "Match required_tool_names/optional_tool_names to registered_tools.name; "
                    "the registered_tools.schema value is the execution-time JSON Schema."
                ),
            }
        )
    return {
        "scope": "Source declarations, not proof of live MCP availability or successful external calls.",
        "registered_tools": records,
        "toolsets": toolsets,
        "framework_filesystem": {
            "GENERAL_WEB": DEEP_AGENT_FILE_TOOLS,
            "CODE": CODE_WORKER_FILESYSTEM_TOOLS,
        },
        "playwright_mcp_allowed_names": sorted(PLAYWRIGHT_ALLOWED_TOOLS),
        "control_tools": {
            "GENERAL": ["report_general_result"],
            "WEB": [
                "publish_worker_progress",
                "submit_for_review",
                "download_web_artifact",
            ],
            "CODE_WORKER": [
                "publish_worker_progress",
                "submit_code_for_review",
                "respond_to_code_review",
                "submit_continued_code_for_review",
            ],
            "CODE_REVIEWER": [
                "request_code_worker_repair",
                "publish_reviewed_candidate",
                "submit_code_review",
            ],
        },
        "framework_delegation": {"WEB": ["task"], "GENERAL": [], "CODE": []},
        "evaluation_only": [
            "appworld_discover",
            "appworld_execute",
            "appworld_verify",
        ],
        "legacy_shell": "ShellToolMiddleware in build_middlewares; current main path has no caller. CODE uses Docker execute.",
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), ensure_ascii=False, indent=2))
