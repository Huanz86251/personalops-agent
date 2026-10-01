"""Checks for Google-primary, Bailian-fallback web_search routing."""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit
from langgraph.types import Command
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict

from tools.web_tools import (
    BRIGHTDATA_REQUEST_URL,
    SEARCH1API_SEARCH_URL,
    WEB_SEARCH_MCP_URL,
    BrightDataSearchError,
    _parse_brightdata_search_result,
    _parse_mcp_search_result,
    _run_bailian_search,
    _run_brightdata_search,
    _search_query_once,
    _run_search1api_wechat,
    _run_search_discovery_batch,
    _read_cached_search_pages,
    web_search,
    web_search_tool,
    WebSearchInput,
)
from workers.progress import merge_web_search_cache
from remote_mcp_http import direct_mcp_http_client


class WebSearchMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_search1api_wechat_uses_bounded_content_request(self):
        observed = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"results": [
                    {"title": "Article 1", "link": "https://mp.weixin.qq.com/s/one",
                     "snippet": "Short", "content": "Full body", "published_date": "2026-09-20"},
                    {"title": "Article 2", "link": "https://mp.weixin.qq.com/s/two",
                     "snippet": "Only snippet", "content": ""},
                ]}

        class FakeClient:
            def __init__(self, **kwargs):
                observed["client"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, **kwargs):
                observed["url"] = url
                observed["request"] = kwargs
                return FakeResponse()

        with (
            patch("tools.web_tools.dotenv_values", return_value={
                "SEARCH1API_API_KEY": "private-search-key",
            }),
            patch("tools.web_tools.httpx.AsyncClient", FakeClient),
        ):
            result = await _run_search1api_wechat("莱西 今日 热点")

        self.assertEqual(observed["url"], SEARCH1API_SEARCH_URL)
        self.assertFalse(observed["client"]["trust_env"])
        self.assertEqual(observed["request"]["json"]["search_service"], "wechat")
        self.assertEqual(observed["request"]["json"]["max_results"], 3)
        self.assertEqual(observed["request"]["json"]["crawl_results"], 3)
        self.assertEqual(result["results"][0]["body"], "Full body")
        self.assertEqual(result["results"][1]["body"], "Only snippet")
        self.assertEqual(result["crawled_pages"], 1)
        self.assertEqual(result["estimated_credits"], 2)

    async def test_wechat_discovery_parallelizes_up_to_three_complementary_queries(self):
        active = 0
        max_active = 0

        async def fake_wechat(query):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {
                "backend": "search1api_wechat",
                "query": query,
                "results": [{
                    "title": f"{query} {index}",
                    "href": f"https://mp.weixin.qq.com/s/{index}",
                    "body": f"正文 {index}",
                    "content_included": True,
                } for index in range(3)],
                "external_requests": 1,
                "estimated_credits": 4,
            }

        with patch("tools.web_tools._run_search1api_wechat", new=fake_wechat):
            payload, cache = await _run_search_discovery_batch(
                "莱西 今日 热点",
                ["莱西 民生 公众号", "莱西 政务 公众号"],
                search_service="wechat",
            )

        self.assertEqual(max_active, 3)
        self.assertEqual(payload["query_count"], 3)
        self.assertEqual(payload["external_requests"], 3)
        self.assertEqual(payload["estimated_credits"], 12)
        self.assertEqual(payload["displayed_results"], 9)
        self.assertEqual(cache, {})

    async def test_bailian_separate_key_is_required_and_model_key_is_not_used(self):
        with patch.dict(os.environ, {
            "WEB_SEARCH_DASHSCOPE_API_KEY": "",
            "DASHSCOPE_API_KEY": "model-key-must-not-be-used",
        }), patch("tools.web_tools.dotenv_values", return_value={}):
            with self.assertRaisesRegex(ValueError, ".env.web_search.local"):
                await _run_bailian_search("python docs", 5)

    async def test_brightdata_is_primary_and_bailian_is_not_called(self):
        primary = [{"title": f"Google {n}", "href": f"https://example.org/g{n}",
                    "body": f"Summary {n}"} for n in range(5)]
        with (
            patch("tools.web_tools._run_brightdata_search", new=AsyncMock(return_value=primary)),
            patch("tools.web_tools._run_bailian_search", new=AsyncMock()) as fallback,
        ):
            result = await web_search("  python  docs ")
        self.assertEqual(result["backend"], "brightdata_google_serp")
        self.assertEqual(result["query"], "python docs")
        self.assertEqual(result["results"], primary)
        self.assertIsNone(result["fallback_from"])
        fallback.assert_not_awaited()

    async def test_discovery_lane_times_out_primary_then_uses_bounded_fallback(self):
        async def slow_primary(*args, **kwargs):
            await asyncio.sleep(0.5)

        fallback_rows = [{
            "title": "Fallback",
            "href": "https://example.org/fallback",
            "body": "Observed",
        }]
        with (
            patch("tools.web_tools.BRIGHTDATA_SEARCH_TIMEOUT_SECONDS", 0.1),
            patch("tools.web_tools.BAILIAN_SEARCH_TIMEOUT_SECONDS", 0.1),
            patch("tools.web_tools.WEB_SEARCH_LANE_BUDGET_SECONDS", 0.25),
            patch("tools.web_tools._run_brightdata_search", new=slow_primary),
            patch(
                "tools.web_tools._run_bailian_search",
                new=AsyncMock(return_value=fallback_rows),
            ) as fallback,
        ):
            result = await _search_query_once("故宫 开放时间")

        self.assertEqual(result["backend"], "bailian_web_search_mcp")
        self.assertEqual(result["external_requests"], 2)
        self.assertIn("超过18秒预算", result["fallback_reason"])
        self.assertIsNotNone(result["timing_ms"]["brightdata"])
        self.assertIsNotNone(result["timing_ms"]["bailian"])
        fallback.assert_awaited_once()

    async def test_discovery_lane_shares_one_budget_across_primary_and_fallback(self):
        async def slow_primary(*args, **kwargs):
            await asyncio.sleep(0.5)

        async def slow_fallback(*args, **kwargs):
            await asyncio.sleep(0.5)

        started = asyncio.get_running_loop().time()
        with (
            patch("tools.web_tools.BRIGHTDATA_SEARCH_TIMEOUT_SECONDS", 0.1),
            patch("tools.web_tools.BAILIAN_SEARCH_TIMEOUT_SECONDS", 0.1),
            patch("tools.web_tools.WEB_SEARCH_LANE_BUDGET_SECONDS", 0.18),
            patch("tools.web_tools._run_brightdata_search", new=slow_primary),
            patch("tools.web_tools._run_bailian_search", new=slow_fallback),
        ):
            result = await _search_query_once("故宫 临时公告")
        elapsed = asyncio.get_running_loop().time() - started

        self.assertEqual(result["backend"], "unavailable")
        self.assertIn("均不可用", result["error"])
        self.assertIn("剩余回退预算", result["fallback_error"])
        self.assertLess(elapsed, 0.3)

    async def test_brightdata_request_uses_google_page_and_private_key(self):
        observed = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"status_code": 200, "body": json.dumps({"organic": [{
                    "title": "Official", "link": "https://example.org/page",
                    "description": "Excerpt",
                }]})}

        class FakeClient:
            def __init__(self, **kwargs):
                observed["client"] = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, **kwargs):
                observed["url"] = url
                observed["request"] = kwargs
                return FakeResponse()

        with (
            patch("tools.web_tools.dotenv_values", return_value={
                "BRIGHTDATA_API_KEY": "private-key", "BRIGHTDATA_ZONE": "serp_api1",
            }),
            patch("tools.web_tools.httpx.AsyncClient", FakeClient),
        ):
            result = await _run_brightdata_search("python docs", 2)
        target = observed["request"]["json"]["url"]
        query = parse_qs(urlsplit(target).query)
        self.assertEqual(observed["url"], BRIGHTDATA_REQUEST_URL)
        self.assertFalse(observed["client"]["trust_env"])
        self.assertEqual(observed["request"]["headers"]["Authorization"], "Bearer private-key")
        self.assertEqual(observed["request"]["json"]["data_format"], "parsed_light")
        self.assertEqual(query["q"], ["python docs"])
        self.assertEqual(query["start"], ["5"])
        self.assertEqual(result[0]["title"], "Official")

    async def test_brightdata_discovery_can_request_ten_without_extra_call(self):
        observed = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"status_code": 200, "body": json.dumps({"organic": [{
                    "title": f"Official {index}",
                    "link": f"https://example.org/{index}",
                    "description": "Excerpt",
                } for index in range(10)]})}

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, **kwargs):
                observed["target"] = kwargs["json"]["url"]
                observed["calls"] = observed.get("calls", 0) + 1
                return FakeResponse()

        with (
            patch("tools.web_tools.dotenv_values", return_value={
                "BRIGHTDATA_API_KEY": "private-key", "BRIGHTDATA_ZONE": "serp_api1",
            }),
            patch("tools.web_tools.httpx.AsyncClient", FakeClient),
        ):
            result = await _run_brightdata_search("python docs", 1, 10)
        query = parse_qs(urlsplit(observed["target"]).query)
        self.assertEqual(observed["calls"], 1)
        self.assertEqual(query["num"], ["10"])
        self.assertEqual(query["start"], ["0"])
        self.assertEqual(len(result), 10)

    async def test_calls_bailian_mcp_and_keeps_existing_result_shape(self):
        observed = {}
        pages = [
            {"title": f"Result {n}", "url": f"https://example.org/{n}",
             "snippet": f"Summary {n}"}
            for n in range(1, 11)
        ]

        @asynccontextmanager
        async def fake_transport(url, **kwargs):
            observed["url"] = url
            observed["headers"] = kwargs["headers"]
            observed["client_factory"] = kwargs["httpx_client_factory"]
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

            async def list_tools(self):
                return SimpleNamespace(tools=[SimpleNamespace(
                    name="bailian_web_search",
                    inputSchema={"properties": {"query": {}, "count": {}}},
                )])

            async def call_tool(self, name, arguments):
                observed["call"] = (name, arguments)
                return SimpleNamespace(isError=False, structuredContent={"pages": pages})

        with (
            patch.dict(os.environ, {"WEB_SEARCH_DASHSCOPE_API_KEY": "search-key"}),
            patch("tools.web_tools.dotenv_values", return_value={}),
            patch("tools.web_tools.streamablehttp_client", fake_transport),
            patch("tools.web_tools.ClientSession", FakeSession),
        ):
            result = await _run_bailian_search("python docs", 10)

        self.assertEqual(observed["url"], WEB_SEARCH_MCP_URL)
        self.assertEqual(observed["headers"], {"Authorization": "Bearer search-key"})
        self.assertIs(observed["client_factory"], direct_mcp_http_client)
        self.assertEqual(observed["call"], (
            "bailian_web_search", {"query": "python docs", "count": 10},
        ))
        self.assertEqual(result[0], {
            "title": "Result 1", "href": "https://example.org/1", "body": "Summary 1",
        })
        self.assertEqual(len(result), 10)

    async def test_primary_failure_falls_back_and_pages_bailian_results(self):
        pages = [{"title": f"Result {n}", "href": f"https://example.org/{n}",
                  "body": f"Summary {n}"} for n in range(1, 11)]
        with (
            patch("tools.web_tools._run_brightdata_search", new=AsyncMock(
                side_effect=BrightDataSearchError("Bright Data Google SERP 连接失败。"))),
            patch("tools.web_tools._run_bailian_search", new=AsyncMock(return_value=pages)),
        ):
            result = await web_search("python docs", page=2)
        self.assertEqual(result["backend"], "bailian_web_search_mcp")
        self.assertEqual(result["fallback_from"], "brightdata_google_serp")
        self.assertEqual(result["results"][0]["title"], "Result 6")
        self.assertEqual(len(result["results"]), 5)

    def test_brightdata_parser_rejects_failed_or_empty_payload(self):
        with self.assertRaises(ValueError):
            _parse_brightdata_search_result({"status_code": 502, "body": ""})
        with self.assertRaises(ValueError):
            _parse_brightdata_search_result({"status_code": 200, "body": '{"organic":[]}'})
        parsed = _parse_brightdata_search_result({"status_code": 200, "body": json.dumps({
            "organic": [{"title": "Official", "link": "https://example.org/page",
                         "description": "Excerpt" * 300}],
        })})
        self.assertEqual(parsed[0]["href"], "https://example.org/page")
        self.assertLessEqual(len(parsed[0]["body"]), 1201)

    def test_mcp_error_and_invalid_payload_are_not_empty_results(self):
        with self.assertRaises(ValueError):
            _parse_mcp_search_result(SimpleNamespace(isError=True, content=[]))
        with self.assertRaises(ValueError):
            _parse_mcp_search_result(SimpleNamespace(isError=False, content=[]))

        parsed = _parse_mcp_search_result(SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(text=json.dumps({"pages": [{
                "title": "Official", "url": "https://example.org/page",
                "snippet": "Excerpt" * 300,
            }]}))],
        ))
        self.assertEqual(parsed[0]["href"], "https://example.org/page")
        self.assertLessEqual(len(parsed[0]["body"]), 1201)

    async def test_transport_exception_cannot_echo_a_credential(self):
        with (
            patch("tools.web_tools._run_brightdata_search", new=AsyncMock(
                side_effect=ValueError("Bearer private-primary-key"))),
            patch("tools.web_tools._run_bailian_search", new=AsyncMock(
                side_effect=ValueError("Bearer private-fallback-key"))),
        ):
            result = await web_search("python docs")
        self.assertIn("均不可用", result["error"])
        self.assertNotIn("private-primary-key", str(result))
        self.assertNotIn("private-fallback-key", str(result))

    async def test_discovery_parallelizes_three_queries_and_caches_all_returned_results(self):
        active = 0
        max_active = 0

        async def fake_search(query):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            slug = query.replace(" ", "-")
            return {
                "backend": "brightdata_google_serp",
                "query": query,
                "results": [{
                    "title": f"{query} {index}",
                    "href": f"https://example.org/{slug}/{index}",
                    "body": f"Summary {index}",
                } for index in range(10)],
                "fallback_from": None,
                "fallback_reason": None,
                "external_requests": 1,
            }

        with patch("tools.web_tools._search_query_once", new=fake_search):
            payload, cache = await _run_search_discovery_batch(
                "main query", ["official query", "review query"],
            )

        self.assertEqual(max_active, 3)
        self.assertEqual(payload["external_requests"], 3)
        self.assertEqual(payload["displayed_results"], 9)
        self.assertEqual(len(payload["groups"]), 3)
        self.assertTrue(all(group["returned_in_upstream_response"] == 10
                            for group in payload["groups"]))
        self.assertTrue(all(group["cached_remaining"] == 7
                            for group in payload["groups"]))
        self.assertEqual(
            [[item["source_ref"] for item in group["results"]]
             for group in payload["groups"]],
            [
                ["WA1", "WA2", "WA3"],
                ["WB1", "WB2", "WB3"],
                ["WC1", "WC2", "WC3"],
            ],
        )
        self.assertEqual(len(cache), 3)
        self.assertTrue(all(len(entry["results"]) == 10 for entry in cache.values()))

    async def test_discovery_rejects_more_than_three_unique_queries(self):
        payload, cache = await _run_search_discovery_batch(
            "one", ["two", "three", "four"],
        )
        self.assertIn("最多搜索3组", payload["error"])
        self.assertEqual(cache, {})

    def test_cached_pagination_uses_actual_response_and_never_calls_remote(self):
        ref = "Q1@abcdef1234"
        cache = {ref: {
            "query_ref": "Q1",
            "source_lane": "WA",
            "query": "exact original query",
            "backend": "brightdata_google_serp",
            "results": [{
                "title": f"Result {index}",
                "href": f"https://example.org/{index}",
                "body": f"Summary {index}",
                "source_ref": f"WA{index + 1}",
            } for index in range(10)],
            "cursor": 3,
        }}

        page_two, updates = _read_cached_search_pages([ref], cache)
        self.assertEqual(page_two["external_requests"], 0)
        self.assertEqual(len(page_two["groups"][0]["results"]), 3)
        self.assertEqual(
            [item["source_ref"] for item in page_two["groups"][0]["results"]],
            ["WA4", "WA5", "WA6"],
        )
        self.assertEqual(page_two["groups"][0]["cached_remaining"], 4)

        merged = merge_web_search_cache(cache, updates)
        page_three, updates = _read_cached_search_pages(
            ["exact original query"], merged,
        )
        self.assertEqual(page_three["external_requests"], 0)
        self.assertEqual(len(page_three["groups"][0]["results"]), 3)
        self.assertEqual(page_three["groups"][0]["cached_remaining"], 1)

        merged = merge_web_search_cache(merged, updates)
        final_page, updates = _read_cached_search_pages([ref], merged)
        self.assertEqual(len(final_page["groups"][0]["results"]), 1)
        self.assertTrue(final_page["groups"][0]["cache_exhausted"])
        self.assertIsNone(final_page["groups"][0]["next_page_ref"])

    def test_web_search_tool_schema_hides_runtime_and_exposes_batch_contract(self):
        self.assertNotIn("runtime", web_search_tool.args)
        self.assertEqual(
            set(web_search_tool.args),
            {"query", "search_service", "alternative_queries", "continue_refs"},
        )
        description = web_search_tool.description
        self.assertIn("餐厅或店铺探店", description)
        self.assertIn("城市旅游攻略", description)
        self.assertIn("主动考虑", description)
        self.assertIn("不能单独证明全网最热", description)

    def test_private_runtime_injection_is_accepted_but_never_model_visible(self):
        runtime = ToolRuntime(
            state={},
            context=None,
            config={},
            stream_writer=lambda _: None,
            tool_call_id="search-call",
            store=None,
            tools=[],
        )
        parsed = WebSearchInput.model_validate({
            "query": "莱西 健身房",
            "runtime": runtime,
        })
        self.assertIs(parsed.runtime, runtime)
        self.assertNotIn("runtime", WebSearchInput.model_json_schema()["properties"])

    async def test_tool_node_injects_runtime_without_pydantic_extra_error(self):
        class SearchState(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            web_search_lane_cursor: int
            web_search_cache: dict

        builder = StateGraph(SearchState)
        builder.add_node("tools", ToolNode([web_search_tool]))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        payload = {
            "mode": "discovery",
            "query_count": 1,
            "groups": [],
            "external_requests": 1,
        }
        with patch(
            "tools.web_tools._run_search_discovery_batch",
            new=AsyncMock(return_value=(payload, {})),
        ):
            result = await graph.ainvoke({
                "messages": [AIMessage(content="", tool_calls=[{
                    "name": "web_search",
                    "args": {"query": "莱西 健身房"},
                    "id": "call-1",
                    "type": "tool_call",
                }])],
                "web_search_lane_cursor": 0,
            })
        self.assertEqual(json.loads(result["messages"][-1].content), payload)

    def test_json_array_string_is_repaired_before_tool_execution(self):
        parsed = WebSearchInput.model_validate({
            "query": "莱西 餐厅",
            "alternative_queries": '["莱西 本地特色餐厅", "莱西 高评分餐厅"]',
        })
        self.assertEqual(
            parsed.alternative_queries,
            ["莱西 本地特色餐厅", "莱西 高评分餐厅"],
        )
        page = WebSearchInput.model_validate({"continue_refs": "Q1@cache"})
        self.assertEqual(page.continue_refs, ["Q1@cache"])

    async def test_runtime_tool_writes_cache_and_returns_one_tool_message(self):
        payload = {
            "mode": "discovery",
            "query_count": 1,
            "groups": [],
            "external_requests": 1,
        }
        updates = {"Q1@cache": {"query": "python docs", "results": [], "cursor": 0}}
        runtime = SimpleNamespace(state={}, tool_call_id="call-1")
        with patch(
            "tools.web_tools._run_search_discovery_batch",
            new=AsyncMock(return_value=(payload, updates)),
        ):
            result = await web_search_tool.coroutine(
                runtime=runtime,
                query="python docs",
            )
        self.assertIsInstance(result, Command)
        self.assertEqual(result.update["web_search_cache"], updates)
        self.assertEqual(result.update["web_search_lane_cursor"], 1)
        self.assertEqual(result.update["messages"][0].tool_call_id, "call-1")
        self.assertEqual(json.loads(result.update["messages"][0].content), payload)

    async def test_runtime_tool_marks_only_concrete_rows_as_citable(self):
        payload = {
            "mode": "discovery",
            "query_count": 1,
            "groups": [{"results": [{
                "source_ref": "WA1",
                "title": "Local result",
                "href": "https://example.com/local",
            }]}],
            "external_requests": 1,
        }
        runtime = SimpleNamespace(state={}, tool_call_id="call-search")
        with patch(
            "tools.web_tools._run_search_discovery_batch",
            new=AsyncMock(return_value=(payload, {})),
        ):
            result = await web_search_tool.coroutine(
                runtime=runtime,
                query="local query",
            )
        visible = json.loads(result.update["messages"][0].content)
        self.assertEqual(
            visible["citation_contract"]["citable_source_refs"],
            ["WA1"],
        )
        self.assertIn("SEARCH_BATCH", visible["citation_contract"]["instruction"])

    async def test_later_discovery_allocates_new_source_lanes(self):
        payload = {
            "mode": "discovery",
            "query_count": 3,
            "groups": [],
            "external_requests": 3,
        }
        runtime = SimpleNamespace(
            state={"web_search_lane_cursor": 3},
            tool_call_id="call-2",
        )
        search = AsyncMock(return_value=(payload, {}))
        with patch("tools.web_tools._run_search_discovery_batch", new=search):
            result = await web_search_tool.coroutine(
                runtime=runtime,
                query="new main query",
                alternative_queries=["new second query", "new third query"],
            )

        self.assertEqual(search.await_args.kwargs["starting_lane"], 4)
        self.assertEqual(result.update["web_search_lane_cursor"], 6)

    def test_search_cache_reducer_keeps_latest_twenty_four_entries(self):
        merged = merge_web_search_cache(
            {f"Q1@{index}": {"cursor": index} for index in range(20)},
            {f"Q2@{index}": {"cursor": index} for index in range(10)},
        )
        self.assertEqual(len(merged), 24)
        self.assertNotIn("Q1@0", merged)
        self.assertIn("Q2@9", merged)

    def test_mcp_http_client_does_not_read_system_proxy(self):
        with patch("remote_mcp_http.httpx.AsyncClient") as client:
            direct_mcp_http_client(headers={"Authorization": "Bearer test"})
        self.assertIs(client.call_args.kwargs["trust_env"], False)


if __name__ == "__main__":
    unittest.main()
