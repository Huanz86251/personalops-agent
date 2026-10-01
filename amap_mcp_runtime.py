"""Bailian Amap Maps MCP lifecycle and its complete tool contract."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from urllib.parse import quote

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from remote_mcp_http import direct_mcp_http_client


logger = logging.getLogger("agent")
AMAP_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"
AMAP_PERSONAL_MCP_URL = "https://mcp.amap.com/mcp"
AMAP_POI_TOOL_NAMES = (
    "maps_text_search",
    "maps_around_search",
    "maps_search_detail",
    "maps_geo",
)

AMAP_ROUTE_TOOL_NAMES = (
    "maps_geo",
    "maps_direction_bicycling",
    "maps_direction_driving",
    "maps_direction_transit_integrated",
    "maps_direction_walking",
    "maps_distance",
)

AMAP_LOCATION_CONTEXT_TOOL_NAMES = (
    "maps_geo",
    "maps_regeocode",
    "maps_ip_location",
    "maps_weather",
)

AMAP_ACTION_LINK_TOOL_NAMES = (
    "maps_schema_personal_map",
    "maps_schema_navi",
    "maps_schema_take_taxi",
)

# The remote MCP is still accepted only when its complete public contract is
# available.  Toolsets below expose smaller, task-coherent views of this pool.
AMAP_TOOL_NAMES = tuple(dict.fromkeys((
    *AMAP_POI_TOOL_NAMES,
    *AMAP_ROUTE_TOOL_NAMES,
    *AMAP_LOCATION_CONTEXT_TOOL_NAMES,
    *AMAP_ACTION_LINK_TOOL_NAMES,
)))


class AmapMapsMCPRuntime:
    """Register Amap only when the remote server exposes all expected tools."""

    def __init__(self, api_key: str, personal_api_key: str = "") -> None:
        self.api_key = api_key.strip()
        self.personal_api_key = personal_api_key.strip()
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[StructuredTool] = []
        self._tool_lock = asyncio.Lock()

    @property
    def tools(self) -> list[StructuredTool]:
        return list(self._tools)

    @property
    def is_started(self) -> bool:
        return self._exit_stack is not None

    def _wrap_tool(self, source_tool) -> StructuredTool:
        async def invoke_mcp_tool(**kwargs):
            async with self._tool_lock:
                return await source_tool.coroutine(**kwargs)

        return StructuredTool.from_function(
            coroutine=invoke_mcp_tool,
            name=source_tool.name,
            description=source_tool.description,
            args_schema=source_tool.args_schema,
            infer_schema=False,
            return_direct=source_tool.return_direct,
            response_format=source_tool.response_format,
            tags=source_tool.tags,
            metadata=source_tool.metadata,
            handle_tool_error=source_tool.handle_tool_error,
            handle_validation_error=source_tool.handle_validation_error,
        )

    async def start(self) -> None:
        if self.is_started or not (self.personal_api_key or self.api_key):
            return

        exit_stack = AsyncExitStack()
        try:
            if self.personal_api_key:
                # The official personal-key endpoint requires the key in its URL.
                # Never log this URL or attach it to tracing metadata.
                endpoint = (
                    AMAP_PERSONAL_MCP_URL
                    + "?key="
                    + quote(self.personal_api_key, safe="")
                )
                transport_options = {}
                source = "personal"
            else:
                endpoint = AMAP_MCP_URL
                transport_options = {
                    "headers": {"Authorization": f"Bearer {self.api_key}"}
                }
                source = "bailian_trial"
            read, write, _ = await exit_stack.enter_async_context(
                streamablehttp_client(
                    endpoint,
                    **transport_options,
                    timeout=20,
                    httpx_client_factory=direct_mcp_http_client,
                )
            )
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            upstream_tools = await load_mcp_tools(session)
            by_name = {tool.name: tool for tool in upstream_tools}
            missing = set(AMAP_TOOL_NAMES) - by_name.keys()
            if missing:
                raise RuntimeError("缺少必要工具：" + ", ".join(sorted(missing)))
            tools = [self._wrap_tool(by_name[name]) for name in AMAP_TOOL_NAMES]
        except Exception as exc:
            try:
                await exit_stack.aclose()
            except Exception:
                logger.warning("高德地图MCP连接清理失败，工具组仍保持隐藏")
            # Avoid logging exception text: transport errors can contain credentials.
            logger.warning("高德地图MCP不可用，工具组已隐藏 | reason=%s", type(exc).__name__)
            return

        self._exit_stack = exit_stack
        self._tools = tools
        logger.info(
            "高德地图MCP已连接 | tools=%d | source=%s",
            len(tools),
            source,
        )

    async def stop(self) -> None:
        exit_stack = self._exit_stack
        if exit_stack is None:
            return
        self._tools = []
        self._exit_stack = None
        await exit_stack.aclose()
