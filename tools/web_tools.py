from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Literal
from uuid import uuid4
from urllib.parse import (
    urlencode,
    urlsplit,
)

import httpx
from dotenv import dotenv_values, load_dotenv
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from path import PROJECT_ROOT
from remote_mcp_http import direct_mcp_http_client
from config import (
    WEB_SEARCH_HARD_MAX_PARALLELISM,
)


# 即使单独运行web_tools.py测试，
# 也能正确读取项目根目录中的.env。
load_dotenv(
    dotenv_path=(
        PROJECT_ROOT
        / ".env"
    ),
    override=False,
)


SEARCH_PAGE_SIZE = 5
SEARCH_MAX_PAGE = 2
SEARCH_SNIPPET_MAX_CHARS = 1200
SEARCH_DISCOVERY_DISPLAY_SIZE = 3
SEARCH_DISCOVERY_MAX_QUERIES = 3
SEARCH_UPSTREAM_RESULT_LIMIT = 10
BRIGHTDATA_REQUEST_URL = "https://api.brightdata.com/request"
BRIGHTDATA_DEFAULT_ZONE = "serp_api1"
WEB_SEARCH_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
WEB_SEARCH_MCP_TOOL = "bailian_web_search"
SEARCH1API_SEARCH_URL = "https://api.search1api.com/search"
SEARCH1API_WECHAT_MAX_RESULTS = 3
SEARCH1API_WECHAT_CRAWL_RESULTS = 3
SEARCH1API_CONTENT_MAX_CHARS = 8000
BRIGHTDATA_SEARCH_TIMEOUT_SECONDS = 18.0
BAILIAN_SEARCH_TIMEOUT_SECONDS = 12.0
WEB_SEARCH_LANE_BUDGET_SECONDS = 30.0
_web_search_max_parallelism = WEB_SEARCH_HARD_MAX_PARALLELISM
_web_search_capacity = asyncio.Semaphore(_web_search_max_parallelism)


def configure_web_search_parallelism(
    max_parallelism: int,
) -> None:
    """在程序接收请求前设置WebSearch进程级并发上限。"""

    if not (
        1
        <= max_parallelism
        <= WEB_SEARCH_HARD_MAX_PARALLELISM
    ):
        raise ValueError(
            "WebSearch并发数必须在"
            f"1到{WEB_SEARCH_HARD_MAX_PARALLELISM}之间。"
        )

    global _web_search_capacity
    global _web_search_max_parallelism

    _web_search_max_parallelism = max_parallelism
    _web_search_capacity = asyncio.Semaphore(max_parallelism)


def get_web_search_max_parallelism(
) -> int:
    """返回当前实际生效的WebSearch并发上限。"""

    return _web_search_max_parallelism


class WebSearchMCPError(ValueError):
    """可安全返回给调用方的本地搜索错误。"""


class BrightDataSearchError(ValueError):
    """不包含响应正文或凭据的Google SERP错误。"""


class Search1APIError(ValueError):
    """不包含响应正文或凭据的微信公众号搜索错误。"""


class WebSearchInput(BaseModel):
    """Stable public contract for batched discovery and cache pagination."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # ToolNode injects this field after the model has emitted its public tool
    # call.  It must be accepted by the private validation model while staying
    # absent from the model-facing JSON schema.  Without the field, the
    # injected value is rejected as an extra input before any search runs.
    runtime: SkipJsonSchema[ToolRuntime] = None

    query: str = Field(
        default="",
        description=(
            "新检索的主关键词。例：莱西 餐厅 推荐。翻页时必须留空。"
        ),
    )
    search_service: Literal["web", "wechat"] = Field(
        default="web",
        description=(
            "搜索范围。默认web使用通用网页搜索；用户明确要微信公众号文章时传wechat。"
            "中文生活服务、餐厅或店铺探店、营业信息、商圈与周边攻略、城市旅游攻略、"
            "本地活动和本地热点等场景，也应主动考虑用wechat补充更贴近本地生活的内容。"
            "法律政策、官方票务、产品规格等仍优先web和官方原文，公众号内容不能单独证明"
            "全网最热、官方规则或实时库存。wechat每组关键词"
            "固定返回最多3条并尽量为3条都附正文；主query与最多2个互补查询"
            "可并发执行。"
        ),
    )
    alternative_queries: list[str] = Field(
        default_factory=list,
        max_length=2,
        description=(
            "最多2个互补关键词，必须是JSON数组，不是包含数组的字符串。"
            "正确：[\"莱西 本地特色餐厅\",\"莱西 高评分餐厅\"]；"
            "错误：\"[\\\"莱西 本地特色餐厅\\\"]\"。"
        ),
    )
    continue_refs: list[str] = Field(
        default_factory=list,
        description=(
            "只用于读取上次搜索的本地缓存页；传返回的next_page_ref或原始query。"
            "使用本字段时query和alternative_queries必须为空。"
        ),
    )

    @field_validator("alternative_queries", "continue_refs", mode="before")
    @classmethod
    def _normalize_json_array_strings(cls, value, info):
        if value is None:
            return []
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return []
        if info.field_name == "continue_refs" and not stripped.startswith("["):
            return [stripped]
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{info.field_name}必须是字符串数组") from error
        if not isinstance(parsed, list):
            raise ValueError(f"{info.field_name}必须是字符串数组")
        return parsed


def _search_local_config() -> dict[str, str]:
    values = dotenv_values(PROJECT_ROOT / ".env.web_search.local")
    return {str(key): str(value or "").strip() for key, value in values.items()}


def _parse_brightdata_search_result(payload: object) -> list[dict[str, str]]:
    """将Bright Data parsed_light organic结果转成稳定的搜索格式。"""
    if not isinstance(payload, dict) or payload.get("status_code") != 200:
        raise BrightDataSearchError("Bright Data Google SERP 返回失败状态。")
    body = payload.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            raise BrightDataSearchError("Bright Data Google SERP 返回格式无法识别。") from None
    if not isinstance(body, dict) or not isinstance(body.get("organic"), list):
        raise BrightDataSearchError("Bright Data Google SERP 返回格式无法识别。")
    results = []
    for item in body["organic"]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("link") or item.get("url") or "").strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        snippet = str(item.get("description") or item.get("snippet") or "").strip()
        if len(snippet) > SEARCH_SNIPPET_MAX_CHARS:
            snippet = snippet[:SEARCH_SNIPPET_MAX_CHARS].rstrip() + "…"
        results.append({"title": str(item.get("title") or "").strip(),
                        "href": url, "body": snippet})
    if not results:
        raise BrightDataSearchError("Bright Data Google SERP 没有返回有效网页结果。")
    return results


def _parse_search1api_result(payload: object) -> tuple[list[dict], int]:
    """Convert Search1API rows to the stable title/href/body result shape."""

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise Search1APIError("Search1API 微信搜索返回格式无法识别。")

    results: list[dict] = []
    crawled_pages = 0
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("link") or item.get("url") or "").strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        content = str(item.get("content") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        content_included = bool(content)
        if content_included:
            crawled_pages += 1
        body = content or snippet
        if len(body) > SEARCH1API_CONTENT_MAX_CHARS:
            body = body[:SEARCH1API_CONTENT_MAX_CHARS].rstrip() + "…"
        result = {
            "title": str(item.get("title") or "").strip(),
            "href": url,
            "body": body,
            "content_included": content_included,
        }
        published_date = str(item.get("published_date") or "").strip()
        if published_date:
            result["published_date"] = published_date
        results.append(result)
    return results, crawled_pages


async def _run_search1api_wechat(query: str) -> dict:
    """Search WeChat articles and include a bounded amount of article text."""

    config = _search_local_config()
    key = (
        config.get("SEARCH1API_API_KEY")
        or os.getenv("SEARCH1API_API_KEY", "")
    ).strip()
    if not key:
        raise Search1APIError(
            "请在 .env.web_search.local 填写 SEARCH1API_API_KEY。"
        )

    try:
        async with _web_search_capacity:
            async with httpx.AsyncClient(
                timeout=WEB_SEARCH_LANE_BUDGET_SECONDS,
                trust_env=False,
            ) as client:
                response = await client.post(
                    SEARCH1API_SEARCH_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "search_service": "wechat",
                        "max_results": SEARCH1API_WECHAT_MAX_RESULTS,
                        "crawl_results": SEARCH1API_WECHAT_CRAWL_RESULTS,
                        "image": False,
                        "include_sites": [],
                        "exclude_sites": [],
                        "language": "auto",
                    },
                )
    except httpx.HTTPError:
        raise Search1APIError("Search1API 微信搜索连接失败。") from None

    if response.status_code != 200:
        raise Search1APIError(
            f"Search1API 微信搜索请求失败（HTTP {response.status_code}）。"
        )
    try:
        payload = response.json()
    except ValueError:
        raise Search1APIError("Search1API 微信搜索返回格式无法识别。") from None

    results, crawled_pages = _parse_search1api_result(payload)
    return {
        "backend": "search1api_wechat",
        "query": query,
        "results": results,
        "external_requests": 1,
        "requested_max_results": SEARCH1API_WECHAT_MAX_RESULTS,
        "requested_crawl_results": SEARCH1API_WECHAT_CRAWL_RESULTS,
        "crawled_pages": crawled_pages,
        "estimated_credits": 1 + crawled_pages,
    }


async def _run_brightdata_search(
    query: str,
    page: int,
    result_count: int = SEARCH_PAGE_SIZE,
) -> list[dict[str, str]]:
    config = _search_local_config()
    key = (config.get("BRIGHTDATA_API_KEY") or os.getenv("BRIGHTDATA_API_KEY", "")).strip()
    zone = (config.get("BRIGHTDATA_ZONE") or os.getenv("BRIGHTDATA_ZONE", "")
            or BRIGHTDATA_DEFAULT_ZONE).strip()
    if not key:
        raise BrightDataSearchError("Bright Data API Key 未配置。")
    if not zone or len(zone) > 100 or not all(char.isalnum() or char in "_-" for char in zone):
        raise BrightDataSearchError("Bright Data zone 配置无效。")
    google_url = "https://www.google.com/search?" + urlencode({
        "q": query, "num": result_count,
        "start": (page - 1) * result_count, "hl": "zh-CN",
    })
    try:
        async with _web_search_capacity:
            async with httpx.AsyncClient(
                timeout=BRIGHTDATA_SEARCH_TIMEOUT_SECONDS,
                trust_env=False,
            ) as client:
                response = await client.post(
                    BRIGHTDATA_REQUEST_URL,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json={"zone": zone, "url": google_url, "format": "json",
                          "data_format": "parsed_light"},
                )
    except httpx.HTTPError:
        raise BrightDataSearchError("Bright Data Google SERP 连接失败。") from None
    if response.status_code != 200:
        raise BrightDataSearchError("Bright Data Google SERP 请求失败。")
    try:
        payload = response.json()
    except ValueError:
        raise BrightDataSearchError("Bright Data Google SERP 返回格式无法识别。") from None
    return _parse_brightdata_search_result(payload)


def _parse_mcp_search_result(result: object) -> list[dict[str, str]]:
    """将百炼 pages 转成原 web_search 的 title/href/body 格式。"""
    if getattr(result, "isError", False):
        raise WebSearchMCPError("百炼 WebSearch MCP 返回工具错误。")

    data = getattr(result, "structuredContent", None)
    if not isinstance(data, dict) or "pages" not in data:
        data = None
        for block in getattr(result, "content", []):
            raw = getattr(block, "text", None)
            if not isinstance(raw, str):
                continue
            try:
                candidate = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "pages" in candidate:
                data = candidate
                break

    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise WebSearchMCPError("百炼 WebSearch MCP 的结果格式无法识别。")

    results = []
    for page in data["pages"]:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or page.get("href") or "").strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        body = str(page.get("snippet") or page.get("body") or "").strip()
        if len(body) > SEARCH_SNIPPET_MAX_CHARS:
            body = body[:SEARCH_SNIPPET_MAX_CHARS].rstrip() + "…"
        results.append({
            "title": str(page.get("title") or "").strip(),
            "href": url,
            "body": body,
        })
    if data["pages"] and not results:
        raise WebSearchMCPError("百炼 WebSearch MCP 返回的结果缺少有效网页链接。")
    return results


async def _run_bailian_search(query: str, count: int) -> list[dict[str, str]]:
    config = _search_local_config()
    key = (
        config.get("WEB_SEARCH_DASHSCOPE_API_KEY")
        or os.getenv("WEB_SEARCH_DASHSCOPE_API_KEY", "")
    ).strip()
    if not key:
        raise WebSearchMCPError("请在 .env.web_search.local 填写 WEB_SEARCH_DASHSCOPE_API_KEY。")

    async with _web_search_capacity:
        async with streamablehttp_client(
            WEB_SEARCH_MCP_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=BAILIAN_SEARCH_TIMEOUT_SECONDS,
            httpx_client_factory=direct_mcp_http_client,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                search_tool = next(
                    (item for item in tools if item.name == WEB_SEARCH_MCP_TOOL),
                    None,
                )
                if search_tool is None:
                    raise WebSearchMCPError("百炼 WebSearch MCP 没有提供 bailian_web_search 工具。")
                schema = search_tool.inputSchema or {}
                properties = schema.get("properties", {})
                if "query" not in properties:
                    raise WebSearchMCPError("百炼 WebSearch MCP 的搜索参数与预期不符。")
                arguments = {"query": query}
                if "count" in properties:
                    arguments["count"] = count
                elif count != SEARCH_PAGE_SIZE:
                    raise WebSearchMCPError("百炼 WebSearch MCP 不支持翻页。")
                response = await session.call_tool(search_tool.name, arguments)
                return _parse_mcp_search_result(response)


async def web_search(
    query: str,
    page: int = 1,
) -> dict:
    """搜索互联网中的公开网页。

    默认通过Bright Data Google SERP搜索；主后端失败或无有效结果时，
    自动回退到百炼WebSearch MCP。每页最多返回5条标题、URL和摘要。
    结果相关但覆盖不足时，保持query并增加page继续搜索；
    结果偏题或质量差时，先调整关键词或限定官方域名。

    本工具只负责搜索，不会打开、读取或点击网页。
    深入读取时，有fetch_webpage可先读取公开静态正文；
    动态或交互页面使用实际可用的浏览器工具。
    除非用户明确要求，否则优先选择可直接阅读的文本网页，而非视频网站。
    国内地名和机构优先中文全名加地区。查询国外网站、海外机构、
    英文产品或国际现状时，请优先使用准确的英文搜索词；
    技术问题保留API名、版本和关键报错。

    Args:
        query: 清晰具体的搜索关键词；查询国外信息时优先使用英文。
        page: 搜索结果页码，从1开始，最多2页。

    Returns:
        实际搜索后端、是否回退、页码和搜索结果。
    """

    normalized_query = (
        " ".join(
            query
            .strip()
            .split()
        )
    )

    if not normalized_query:
        return {
            "error": (
                "query不能为空。"
            )
        }

    if page < 1:
        return {
            "error": (
                "page必须从1开始。"
            )
        }

    if page > SEARCH_MAX_PAGE:
        return {"error": f"page最多为{SEARCH_MAX_PAGE}。"}

    fallback_reason = None
    try:
        results = await _run_brightdata_search(normalized_query, page)
        backend = "brightdata_google_serp"
    except Exception as primary_error:
        fallback_reason = (str(primary_error) if isinstance(primary_error, BrightDataSearchError)
                           else f"Bright Data Google SERP 调用失败（{type(primary_error).__name__}）。")
        try:
            all_results = await _run_bailian_search(
                normalized_query, page * SEARCH_PAGE_SIZE,
            )
            results = all_results[(page - 1) * SEARCH_PAGE_SIZE:page * SEARCH_PAGE_SIZE]
            if not results:
                raise WebSearchMCPError("百炼 WebSearch MCP 没有返回有效网页结果。")
            backend = "bailian_web_search_mcp"
        except Exception as fallback_error:
            safe_fallback = (str(fallback_error) if isinstance(fallback_error, WebSearchMCPError)
                             else f"百炼 WebSearch MCP 调用失败（{type(fallback_error).__name__}）。")
            return {"error": "Google主搜索和百炼备用搜索均不可用。",
                    "primary_error": fallback_reason, "fallback_error": safe_fallback,
                    "backend": "unavailable", "query": normalized_query, "page": page}

    return {
        "backend": backend,

        "query": normalized_query,

        "page": page,

        "results": results,

        # 返回完整5条时，
        # 提示模型可以继续请求下一页。
        "fallback_from": "brightdata_google_serp" if fallback_reason else None,
        "fallback_reason": fallback_reason,
        "next_page": page + 1 if len(results) >= SEARCH_PAGE_SIZE and page < SEARCH_MAX_PAGE else None,
    }


def _normalize_search_query(value: str) -> str:
    return " ".join(str(value or "").strip().split())


async def _search_query_once(query: str) -> dict:
    """Run one remote search and retain every valid result in that response."""

    started = time.monotonic()
    timing_ms: dict[str, int | None] = {
        "brightdata": None,
        "bailian": None,
        "total": None,
    }
    fallback_reason = None
    primary_started = time.monotonic()
    try:
        results = await asyncio.wait_for(
            _run_brightdata_search(
                query,
                1,
                SEARCH_UPSTREAM_RESULT_LIMIT,
            ),
            timeout=BRIGHTDATA_SEARCH_TIMEOUT_SECONDS,
        )
        timing_ms["brightdata"] = round(
            (time.monotonic() - primary_started) * 1000
        )
        backend = "brightdata_google_serp"
        external_requests = 1
    except Exception as primary_error:
        timing_ms["brightdata"] = round(
            (time.monotonic() - primary_started) * 1000
        )
        fallback_reason = (
            str(primary_error)
            if isinstance(primary_error, BrightDataSearchError)
            else (
                "Bright Data Google SERP 超过18秒预算。"
                if isinstance(primary_error, TimeoutError)
                else f"Bright Data Google SERP 调用失败（{type(primary_error).__name__}）。"
            )
        )
        elapsed = time.monotonic() - started
        remaining = max(0.0, WEB_SEARCH_LANE_BUDGET_SECONDS - elapsed)
        fallback_timeout = min(BAILIAN_SEARCH_TIMEOUT_SECONDS, remaining)
        if fallback_timeout <= 0:
            timing_ms["total"] = round(elapsed * 1000)
            return {
                "error": "Google主搜索已耗尽本分支30秒预算，未启动百炼备用搜索。",
                "primary_error": fallback_reason,
                "fallback_error": "本分支没有剩余回退预算。",
                "backend": "unavailable",
                "query": query,
                "results": [],
                "external_requests": 1,
                "timing_ms": timing_ms,
            }
        fallback_started = time.monotonic()
        try:
            results = await asyncio.wait_for(
                _run_bailian_search(
                    query,
                    SEARCH_UPSTREAM_RESULT_LIMIT,
                ),
                timeout=fallback_timeout,
            )
            timing_ms["bailian"] = round(
                (time.monotonic() - fallback_started) * 1000
            )
            if not results:
                raise WebSearchMCPError("百炼 WebSearch MCP 没有返回有效网页结果。")
            backend = "bailian_web_search_mcp"
            external_requests = 2
        except Exception as fallback_error:
            timing_ms["bailian"] = round(
                (time.monotonic() - fallback_started) * 1000
            )
            safe_fallback = (
                str(fallback_error)
                if isinstance(fallback_error, WebSearchMCPError)
                else (
                    "百炼 WebSearch MCP 超过剩余回退预算。"
                    if isinstance(fallback_error, TimeoutError)
                    else f"百炼 WebSearch MCP 调用失败（{type(fallback_error).__name__}）。"
                )
            )
            timing_ms["total"] = round(
                (time.monotonic() - started) * 1000
            )
            return {
                "error": "Google主搜索和百炼备用搜索均不可用。",
                "primary_error": fallback_reason,
                "fallback_error": safe_fallback,
                "backend": "unavailable",
                "query": query,
                "results": [],
                "external_requests": 2,
                "timing_ms": timing_ms,
            }

    timing_ms["total"] = round((time.monotonic() - started) * 1000)
    return {
        "backend": backend,
        "query": query,
        # Do not slice here. Some SERP responses return more results than the
        # requested count; all already-paid-for results belong in local cache.
        "results": results,
        "fallback_from": "brightdata_google_serp" if fallback_reason else None,
        "fallback_reason": fallback_reason,
        "external_requests": external_requests,
        "timing_ms": timing_ms,
    }


def _normalize_discovery_queries(
    query: str,
    alternative_queries: list[str] | None,
) -> list[str]:
    candidates = [query, *(alternative_queries or [])]
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = _normalize_search_query(candidate)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            normalized.append(value)
    return normalized


def _search_lane_label(index: int) -> str:
    """Return A..Z, AA.. for stable Worker-visible result references."""

    if index < 1:
        raise ValueError("search lane index must start at 1")
    value = index
    letters = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return "W" + letters


async def _run_search_discovery_batch(
    query: str,
    alternative_queries: list[str] | None = None,
    starting_lane: int = 1,
    search_service: Literal["web", "wechat"] = "web",
) -> tuple[dict, dict[str, dict]]:
    """Search up to three complementary queries concurrently."""

    queries = _normalize_discovery_queries(query, alternative_queries)
    if not queries:
        return {"error": "query不能为空。"}, {}
    if len(queries) > SEARCH_DISCOVERY_MAX_QUERIES:
        return {
            "error": (
                "一次最多搜索3组互补关键词：1个query加最多2个alternative_queries。"
            )
        }, {}

    runner = _run_search1api_wechat if search_service == "wechat" else _search_query_once

    batch_started = time.monotonic()
    responses = await asyncio.gather(*(
        runner(item)
        for item in queries
    ))
    cache_updates: dict[str, dict] = {}
    groups: list[dict] = []

    for index, response in enumerate(responses, start=1):
        query_ref = f"Q{index}"
        source_lane = _search_lane_label(starting_lane + index - 1)
        current_query = queries[index - 1]
        if response.get("error"):
            groups.append({
                "query_ref": query_ref,
                "source_lane": source_lane,
                "query": current_query,
                "backend": response.get("backend", "unavailable"),
                "error": response["error"],
                "primary_error": response.get("primary_error"),
                "fallback_error": response.get("fallback_error"),
                "results": [],
                "returned_in_upstream_response": 0,
                "cached_remaining": 0,
                "next_page_ref": None,
                "timing_ms": response.get("timing_ms"),
            })
            continue

        all_results = [
            {**item, "source_ref": f"{source_lane}{result_index}"}
            for result_index, item in enumerate(response.get("results") or [], start=1)
        ]
        shown = all_results[:SEARCH_DISCOVERY_DISPLAY_SIZE]
        cursor = len(shown)

        continuation_ref = f"{query_ref}@{uuid4().hex[:10]}"
        remaining = max(0, len(all_results) - cursor)
        if remaining:
            cache_updates[continuation_ref] = {
                "query_ref": query_ref,
                "source_lane": source_lane,
                "query": current_query,
                "backend": response["backend"],
                "fallback_from": response.get("fallback_from"),
                "fallback_reason": response.get("fallback_reason"),
                "results": all_results,
                "cursor": cursor,
            }

        groups.append({
            "query_ref": query_ref,
            "source_lane": source_lane,
            "query": current_query,
            "search_service": search_service,
            "backend": response["backend"],
            "fallback_from": response.get("fallback_from"),
            "fallback_reason": response.get("fallback_reason"),
            "results": shown,
            "returned_in_upstream_response": len(all_results),
            "shown_now": len(shown),
            "cached_remaining": remaining,
            "next_page_ref": continuation_ref if remaining else None,
            "timing_ms": response.get("timing_ms"),
        })

    return {
        "mode": "discovery",
        "search_service": search_service,
        "query_count": len(queries),
        "display_limit_per_query": SEARCH_DISCOVERY_DISPLAY_SIZE,
        "displayed_results": sum(len(group.get("results") or []) for group in groups),
        "external_requests": sum(
            int(response.get("external_requests") or 0)
            for response in responses
        ),
        "estimated_credits": sum(
            int(response.get("estimated_credits") or 0)
            for response in responses
        ) or None,
        "elapsed_ms": round((time.monotonic() - batch_started) * 1000),
        "groups": groups,
        "pagination_contract": (
            "Q1/Q2/Q3只标识查询分支，不是证据编号。需要更多结果时，"
            "把next_page_ref原样传给continue_refs；也可传该组完整原始query。"
            "缓存耗尽前翻页不会再次请求搜索服务。schema引用具体结果的source_ref"
            "（如WA1），不要用Q编号；普通工具仍使用E编号。"
        ),
    }, cache_updates


def _resolve_cached_search_ref(
    requested: str,
    cache: dict[str, dict],
) -> str | None:
    value = str(requested or "").strip()
    if value in cache:
        return value
    normalized = _normalize_search_query(value).casefold()
    exhausted_match = None
    for cache_ref, entry in reversed(list(cache.items())):
        if _normalize_search_query(entry.get("query", "")).casefold() == normalized:
            exhausted_match = cache_ref
            if int(entry.get("cursor") or 0) < len(entry.get("results") or []):
                return cache_ref
    return exhausted_match


def _read_cached_search_pages(
    continue_refs: list[str],
    cache: dict[str, dict],
) -> tuple[dict, dict[str, dict]]:
    """Read the next three results per lane without any remote request."""

    requested = [str(value or "").strip() for value in continue_refs if str(value or "").strip()]
    if not requested:
        return {"error": "continue_refs不能为空。"}, {}
    if len(requested) > SEARCH_DISCOVERY_MAX_QUERIES:
        return {"error": "一次最多继续3个查询分支。"}, {}

    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    unresolved: list[str] = []
    for value in requested:
        cache_ref = _resolve_cached_search_ref(value, cache)
        if cache_ref is None:
            unresolved.append(value)
        elif cache_ref not in seen:
            seen.add(cache_ref)
            resolved.append((value, cache_ref))

    cache_updates: dict[str, dict] = {}
    groups: list[dict] = []
    for requested_value, cache_ref in resolved:
        entry = dict(cache[cache_ref])
        results = list(entry.get("results") or [])
        start = max(0, int(entry.get("cursor") or 0))
        end = min(len(results), start + SEARCH_DISCOVERY_DISPLAY_SIZE)
        page_results = results[start:end]
        entry["cursor"] = end
        cache_updates[cache_ref] = entry
        remaining = max(0, len(results) - end)
        groups.append({
            "requested_ref": requested_value,
            "query_ref": entry.get("query_ref"),
            "source_lane": entry.get("source_lane"),
            "query": entry.get("query"),
            "backend": entry.get("backend"),
            "results": page_results,
            "shown_now": len(page_results),
            "cached_remaining": remaining,
            "cache_exhausted": remaining == 0,
            "next_page_ref": cache_ref if remaining else None,
        })

    payload = {
        "mode": "cached_page",
        "cache_hit": bool(groups),
        "external_requests": 0,
        "groups": groups,
        "unresolved_refs": unresolved,
        "pagination_contract": (
            "本次只读取本地缓存，没有调用搜索服务。next_page_ref为空表示该次"
            "上游响应已读完；如仍缺证据，请换更精确关键词发起新的discovery。"
        ),
    }
    if not groups:
        payload["error"] = "没有找到可继续读取的搜索缓存。"
    return payload, cache_updates


def _web_search_command(
    payload: dict,
    runtime: ToolRuntime,
    cache_updates: dict[str, dict] | None = None,
    state_updates: dict | None = None,
) -> Command:
    payload = dict(payload)
    source_refs: list[str] = []

    def collect_source_refs(item) -> None:
        if isinstance(item, dict):
            source_ref = str(item.get("source_ref") or "").strip()
            if source_ref and source_ref not in source_refs:
                source_refs.append(source_ref)
            for nested in item.values():
                collect_source_refs(nested)
        elif isinstance(item, list):
            for nested in item:
                collect_source_refs(nested)

    collect_source_refs(payload)
    if source_refs:
        payload["citation_contract"] = {
            "citable_source_refs": source_refs,
            "instruction": (
                "事实交接只引用具体source_ref（如WA1/WB2）；"
                "SEARCH_BATCH只是工具调用批次，不是证据，不能写入evidence_refs。"
            ),
        }
    update: dict = {
        "messages": [ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=str(runtime.tool_call_id or "web_search"),
        )]
    }
    if cache_updates:
        update["web_search_cache"] = cache_updates
    if state_updates:
        update.update(state_updates)
    return Command(update=update)


async def _runtime_web_search(
    runtime: ToolRuntime,
    query: str = "",
    search_service: Literal["web", "wechat"] = "web",
    alternative_queries: list[str] | None = None,
    continue_refs: list[str] | None = None,
) -> Command:
    """Runtime entry point for batched discovery and cache-only pagination."""

    if continue_refs:
        if _normalize_search_query(query) or alternative_queries:
            return _web_search_command({
                "error": "翻页时只传continue_refs；新搜索时只传query和alternative_queries。"
            }, runtime)
        payload, updates = _read_cached_search_pages(
            continue_refs,
            dict(runtime.state.get("web_search_cache") or {}),
        )
        return _web_search_command(payload, runtime, updates)

    starting_lane = max(int(runtime.state.get("web_search_lane_cursor") or 0) + 1, 1)
    payload, updates = await _run_search_discovery_batch(
        query,
        alternative_queries,
        starting_lane=starting_lane,
        search_service=search_service,
    )
    query_count = int(payload.get("query_count") or 0)
    state_updates = (
        {"web_search_lane_cursor": starting_lane + query_count - 1}
        if query_count
        else None
    )
    return _web_search_command(payload, runtime, updates, state_updates)


def _parse_github_repository(
    github_url: str,
) -> tuple[
    str,
    str,
    str,
]:
    """解析GitHub仓库URL。"""

    normalized_url = (
        github_url
        .strip()
    )

    if not normalized_url:
        raise ValueError(
            "github_url不能为空。"
        )

    # 同时支持：
    # github.com/owner/repository
    if "://" not in normalized_url:
        normalized_url = (
            "https://"
            + normalized_url
        )

    parsed_url = urlsplit(
        normalized_url
    )

    hostname = (
        parsed_url.hostname
        or ""
    ).lower()

    if hostname not in {
        "github.com",
        "www.github.com",
    }:
        raise ValueError(
            "只支持github.com仓库地址。"
        )

    path_parts = [
        part
        for part in (
            parsed_url.path
            .strip("/")
            .split("/")
        )
        if part
    ]

    if len(path_parts) < 2:
        raise ValueError(
            "GitHub地址必须包含"
            "owner和repository。"
        )

    owner = path_parts[0]

    repository = (
        path_parts[1]
        .removesuffix(
            ".git"
        )
    )

    original_url = (
        "https://github.com/"
        f"{owner}/{repository}"
    )

    return (
        owner,
        repository,
        original_url,
    )

def _build_gitcode_mirror_url(
    repository: str,
) -> str:
    """根据GitCode常见规则生成镜像候选URL。"""

    prefix = (
        repository[:2]
        .casefold()
    )

    return (
        "https://gitcode.com/"
        "gh_mirrors/"
        f"{prefix}/"
        f"{repository}"
    )

async def _verify_gitcode_page(
    url: str,
    owner: str,
    repository: str,
) -> bool:
    """确认GitCode镜像页面可以访问且包含仓库信息。"""

    try:
        async with httpx.AsyncClient(
            timeout=10,

            follow_redirects=True,

            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64)"
                ),
            },
        ) as client:
            response = await client.get(
                url
            )

    except Exception:
        return False

    if response.status_code != 200:
        return False

    page_text = (
        response.text
        .casefold()
    )

    invalid_markers = (
        "404 not found",
        "页面不存在",
        "项目不存在",
        "仓库不存在",
    )

    if any(
        marker in page_text
        for marker in invalid_markers
    ):
        return False

    repository_name = (
        repository
        .casefold()
    )

    original_signature = (
        "github.com/"
        f"{owner}/{repository}"
    ).casefold()

    return (
        repository_name in page_text
        or original_signature in page_text
    )

async def find_github_mirror(
    github_url: str,
) -> dict:
    """查找适合中国网络访问的GitHub仓库镜像。

    工具会根据GitCode常见镜像路径生成候选地址，
    然后真实访问该页面进行基础验证。

    找到时返回GitCode镜像URL；
    不存在时返回not_found。

    镜像只适合公开仓库的只读浏览。

    Args:
        github_url: GitHub公开仓库URL。

    Returns:
        原始仓库地址、镜像地址和验证状态。
    """

    try:
        (
            owner,
            repository,
            original_url,
        ) = _parse_github_repository(
            github_url
        )

    except ValueError as error:
        return {
            "status": "invalid",

            "error": str(
                error
            ),
        }

    mirror_url = (
        _build_gitcode_mirror_url(
            repository
        )
    )

    is_valid = await (
        _verify_gitcode_page(
            mirror_url,

            owner,

            repository,
        )
    )

    if not is_valid:
        return {
            "status": "not_found",

            "github_url": (
                original_url
            ),

            "mirror_url_checked": (
                mirror_url
            ),

            "message": (
                "没有找到能够验证的"
                "GitCode镜像。"
            ),
        }

    return {
        "status": "found",

        "github_url": (
            original_url
        ),

        "mirror_url": (
            mirror_url
        ),

        "note": (
            "镜像页面已经通过基础可用性验证。"
        ),
    }

web_search_tool = tool(
    "web_search",
    args_schema=WebSearchInput,
    description=(
        "搜索公开网页及微信公众号文章。默认search_service='web'；用户明确要求公众号"
        "文章时用search_service='wechat'。对于中文生活服务、餐厅或店铺探店、营业信息、"
        "商圈和周边攻略、城市旅游攻略、本地活动、本地热点等任务，也应主动考虑增加一次"
        "wechat搜索，以取得更贴近当地生活的内容。法律政策、官方票务、产品规格仍应优先"
        "web和官方原文；公众号文章不能单独证明全网最热、官方规则或实时库存。微信模式"
        "每组关键词固定最多3条结果，并尽量让3条都附正文；主query与最多2个"
        "alternative_queries会并发执行，最多三路。每一路成功搜索固定1 credit，"
        "每篇成功正文再加1 credit，因此单路最多4 credits、三路最多12 credits。"
        "只有确有不同检索角度时才增加互补查询，不要用近义改写浪费费用。"
        "通用网页新检索也可用alternative_queries补充最多2组互补关键词；"
        "alternative_queries必须是真正的JSON数组，不能把整个数组再包成字符串。正例："
        "query='莱西 餐厅 推荐', alternative_queries=['莱西 本地特色餐厅','莱西 高评分餐厅']。"
        "反例：连续发起3次只有同义改写的搜索，或传alternative_queries='[\\\"词一\\\"]'。"
        "Harness会在一次工具调用中并行搜索，每组首屏只显示3条，但会缓存这次上游响应"
        "实际返回的全部其余结果。优先一次提交互补查询，不要串行重复调用同义搜索。需要更多结果时，"
        "把返回的next_page_ref原样放进continue_refs（也可传完整原始query）；翻页只读"
        "本地缓存且不发新网络请求。Q1/Q2/Q3只用于翻页；每条结果都有WA1、WB2等"
        "source_ref，提交schema时引用具体source_ref。成功搜索的父调用只显示为"
        "SEARCH_BATCH，不会获得E证据编号；它不是证据，不能写入evidence_refs。"
        "只有搜索失败且没有任何具体结果时，整个失败调用才可按显示的E编号证明失败。"
        "普通工具结果仍引用E编号。"
    ),
)(_runtime_web_search)


github_mirror_tool = tool(
    parse_docstring=True
)(find_github_mirror)
