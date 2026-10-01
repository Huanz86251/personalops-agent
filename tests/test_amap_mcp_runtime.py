"""Offline checks for Amap MCP availability and Toolset registration."""

import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from amap_mcp_runtime import (
    AMAP_MCP_URL,
    AMAP_PERSONAL_MCP_URL,
    AMAP_POI_TOOL_NAMES,
    AMAP_ROUTE_TOOL_NAMES,
    AMAP_LOCATION_CONTEXT_TOOL_NAMES,
    AMAP_ACTION_LINK_TOOL_NAMES,
    AMAP_TOOL_NAMES,
    AmapMapsMCPRuntime,
)
from remote_mcp_http import direct_mcp_http_client
from toolsets import DEFAULT_TOOLSET_REGISTRY


class EmptyInput(BaseModel):
    pass


async def _call(**kwargs):
    return kwargs


def _source(name):
    return SimpleNamespace(
        name=name, description=name, args_schema=EmptyInput,
        coroutine=_call, return_direct=False, response_format="content",
        tags=[], metadata={}, handle_tool_error=False,
        handle_validation_error=False,
    )


class AmapMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_remote_registers_one_nonexclusive_group(self):
        observed = {}

        @asynccontextmanager
        async def fake_transport(url, **kwargs):
            observed.update(url=url, **kwargs)
            yield object(), object(), None

        class FakeSession:
            def __init__(self, read, write):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def initialize(self):
                pass

        async def fake_load(session):
            return [_source(name) for name in AMAP_TOOL_NAMES]

        runtime = AmapMapsMCPRuntime("test-key")
        with (patch("amap_mcp_runtime.streamablehttp_client", fake_transport),
              patch("amap_mcp_runtime.ClientSession", FakeSession),
              patch("amap_mcp_runtime.load_mcp_tools", fake_load)):
            await runtime.start()
            self.assertTrue(runtime.is_started)
            self.assertEqual([t.name for t in runtime.tools], list(AMAP_TOOL_NAMES))
            catalog = DEFAULT_TOOLSET_REGISTRY.build_router_metadata(runtime.tools)
            names = {entry["name"] for entry in catalog}
            self.assertTrue({
                "AMAP_POI", "AMAP_ROUTE", "AMAP_LOCATION_CONTEXT",
                "AMAP_ACTION_LINKS",
            } <= names)
            for group, expected in (
                ("AMAP_POI", AMAP_POI_TOOL_NAMES),
                ("AMAP_ROUTE", AMAP_ROUTE_TOOL_NAMES),
                ("AMAP_LOCATION_CONTEXT", AMAP_LOCATION_CONTEXT_TOOL_NAMES),
                ("AMAP_ACTION_LINKS", AMAP_ACTION_LINK_TOOL_NAMES),
            ):
                resolved = DEFAULT_TOOLSET_REGISTRY.resolve(group, runtime.tools)
                self.assertTrue(resolved.is_available)
                self.assertFalse(resolved.spec.exclusive)
                self.assertEqual(set(resolved.tool_names), set(expected))
            await runtime.stop()

        self.assertEqual(observed["url"], AMAP_MCP_URL)
        self.assertEqual(observed["headers"], {"Authorization": "Bearer test-key"})
        self.assertIs(observed["httpx_client_factory"], direct_mcp_http_client)
        self.assertFalse(runtime.tools)

    async def test_missing_one_tool_hides_entire_group(self):
        @asynccontextmanager
        async def fake_transport(url, **kwargs):
            yield object(), object(), None

        class FakeSession:
            def __init__(self, read, write):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def initialize(self):
                pass

        async def fake_load(session):
            return [_source(name) for name in AMAP_TOOL_NAMES[:-1]]

        runtime = AmapMapsMCPRuntime("test-key")
        with (patch("amap_mcp_runtime.streamablehttp_client", fake_transport),
              patch("amap_mcp_runtime.ClientSession", FakeSession),
              patch("amap_mcp_runtime.load_mcp_tools", fake_load)):
            await runtime.start()

        self.assertFalse(runtime.is_started)
        self.assertFalse(runtime.tools)
        catalog = DEFAULT_TOOLSET_REGISTRY.build_router_metadata(runtime.tools)
        self.assertFalse({
            "AMAP_POI", "AMAP_ROUTE", "AMAP_LOCATION_CONTEXT",
            "AMAP_ACTION_LINKS",
        } & {entry["name"] for entry in catalog})

    async def test_personal_key_takes_precedence_without_bailian_header(self):
        observed = {}

        @asynccontextmanager
        async def fake_transport(url, **kwargs):
            observed.update(url=url, **kwargs)
            yield object(), object(), None

        class FakeSession:
            def __init__(self, read, write):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def initialize(self):
                pass

        async def fake_load(session):
            return [_source(name) for name in AMAP_TOOL_NAMES]

        runtime = AmapMapsMCPRuntime(
            "bailian-key-must-not-be-used",
            personal_api_key="personal key/+",
        )
        with (patch("amap_mcp_runtime.streamablehttp_client", fake_transport),
              patch("amap_mcp_runtime.ClientSession", FakeSession),
              patch("amap_mcp_runtime.load_mcp_tools", fake_load)):
            await runtime.start()
            await runtime.stop()

        self.assertEqual(
            observed["url"],
            AMAP_PERSONAL_MCP_URL + "?key=personal%20key%2F%2B",
        )
        self.assertNotIn("headers", observed)
        self.assertIs(observed["httpx_client_factory"], direct_mcp_http_client)

    async def test_missing_key_hides_group_without_connection(self):
        runtime = AmapMapsMCPRuntime("")
        with patch("amap_mcp_runtime.streamablehttp_client") as transport:
            await runtime.start()
        transport.assert_not_called()
        self.assertFalse(runtime.tools)

    async def test_connection_failure_hides_group_without_stopping_startup(self):
        @asynccontextmanager
        async def failing_transport(url, **kwargs):
            raise OSError("unavailable")
            yield

        runtime = AmapMapsMCPRuntime("test-key")
        with patch("amap_mcp_runtime.streamablehttp_client", failing_transport):
            await runtime.start()
        self.assertFalse(runtime.is_started)
        self.assertFalse(runtime.tools)


if __name__ == "__main__":
    unittest.main()
