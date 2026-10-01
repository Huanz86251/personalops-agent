"""One read-only 1688 sourcing tool backed by an Aliyun Marketplace API."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from langchain.tools import tool

from path import PROJECT_ROOT


MARKET_1688_URL = "https://cbu.market.alicloudapi.com/deepsearch"
MARKET_1688_CONFIG = PROJECT_ROOT / ".env.1688.local"
_MAX_STREAM_BYTES = 2 * 1024 * 1024
_NON_PRODUCT_TITLE = re.compile(r"邮费|运费|补差|补拍|补款|差价|定金|专拍|[一1]元链接")


def _credentials() -> tuple[str, str]:
    values = dotenv_values(MARKET_1688_CONFIG)
    appcode = str(values.get("ALIYUN_MARKET_1688_APPCODE") or os.getenv("ALIYUN_MARKET_1688_APPCODE", "")).strip()
    app_id = str(values.get("ALIYUN_MARKET_1688_APP_ID") or os.getenv("ALIYUN_MARKET_1688_APP_ID", "")).strip()
    return appcode, app_id


def market_1688_is_configured() -> bool:
    return all(_credentials())


def _as_positive_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _normalize_offer(item: dict) -> dict | None:
    offer = item.get("offerInfo")
    if not isinstance(offer, dict):
        return None
    title = str(offer.get("title") or "").strip()
    if not title or "\ufffd" in title or _NON_PRODUCT_TITLE.search(title):
        return None
    url = str(offer.get("pcDetailUrl") or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "detail.1688.com" or not parsed.path.startswith("/offer/"):
        return None

    tiers = []
    for raw in offer.get("offerPriceRanges") or []:
        if not isinstance(raw, dict):
            continue
        price = _as_positive_number(raw.get("price"))
        try:
            min_quantity = int(raw.get("beginAmount"))
        except (TypeError, ValueError):
            continue
        if price is None or min_quantity < 1:
            continue
        tiers.append({"min_quantity": min_quantity, "unit_price_cny": price})
    tiers.sort(key=lambda tier: tier["min_quantity"])
    if not tiers:
        return None

    try:
        declared_minimum = int(offer.get("minPurchaseCnt"))
    except (TypeError, ValueError):
        declared_minimum = tiers[0]["min_quantity"]
    minimum_order_quantity = max(1, declared_minimum, tiers[0]["min_quantity"])
    entry_tier = max(
        (tier for tier in tiers if tier["min_quantity"] <= minimum_order_quantity),
        key=lambda tier: tier["min_quantity"],
    )

    seller = offer.get("sellerInfo") or {}
    return {
        "title": title[:120],
        "url": url,
        "minimum_order_quantity": minimum_order_quantity,
        "entry_unit_price_cny": entry_tier["unit_price_cny"],
        "price_tiers": tiers[:4],
        "seller": str(seller.get("companyName") or "")[:120] if isinstance(seller, dict) else "",
    }


def _final_offers(event: dict) -> list[dict]:
    """Find the provider's returning ANALYZE data inside its nested SSE envelope."""
    pending = [event]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            response_data = value.get("responseData")
            if isinstance(response_data, dict) and response_data.get("status") == "returning":
                items = response_data.get("data")
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
            for name in ("output", "content", "data"):
                nested = value.get(name)
                if isinstance(nested, (dict, list)):
                    pending.append(nested)
    raise ValueError("1688 接口未返回可识别的商品清单。")


async def _search_once(query: str, appcode: str, app_id: str) -> list[dict]:
    payload = {
        "app_id": app_id,
        "input": [{"role": "user", "content": [{"type": "text", "text": query}]}],
    }
    completed = None
    try:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=True, timeout=60) as client:
            async with client.stream(
                "POST", MARKET_1688_URL,
                headers={"Authorization": f"APPCODE {appcode}", "Content-Type": "application/json", "Accept": "text/event-stream"},
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raise ValueError(f"1688 云市场接口返回 HTTP {response.status_code}。")
                consumed = 0
                async for line in response.aiter_lines():
                    consumed += len(line.encode("utf-8"))
                    if consumed > _MAX_STREAM_BYTES:
                        raise ValueError("1688 云市场返回内容超过大小上限。")
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error") or event.get("status") in {"failed", "error"}:
                        raise ValueError("1688 云市场搜索未成功完成。")
                    if event.get("status") == "completed":
                        completed = event
    except (httpx.HTTPError, TimeoutError) as exc:
        # Do not return the exception string: request errors may include credentials.
        raise ValueError(f"1688 云市场连接失败（{type(exc).__name__}）。") from None

    if completed is None:
        raise ValueError("1688 云市场未返回完成事件。")
    return _final_offers(completed)


async def search_1688_products(
    query: str,
    max_items: int = 20,
    related_queries: list[str] | None = None,
    max_minimum_order_quantity: int = 10,
) -> dict:
    """Search 1688 B2B listings. Up to three related searches may yield 20-30 links.

    Single API calls return at most ten analyzed offers. For a broad sample,
    provide up to two distinct positive related_queries for the same product;
    never put excluded terms such as shipping-fee links into the query text.
    The actual count can be lower after filtering. These are wholesale price
    tiers and minimum order quantities, not Taobao retail or market averages.
    """
    normalized = " ".join(query.split()).strip()
    if not normalized:
        return {"error": "请提供具体商品或选品需求。"}
    if not 1 <= max_items <= 30:
        return {"error": "max_items 必须在 1 到 30 之间。"}
    if not 1 <= max_minimum_order_quantity <= 100000:
        return {"error": "max_minimum_order_quantity 必须在 1 到 100000 之间。"}
    if related_queries is not None and len(related_queries) > 2:
        return {"error": "related_queries 最多填写两个同类商品的不同搜索表达。"}
    appcode, app_id = _credentials()
    if not appcode or not app_id:
        return {"error": "1688 云市场凭证未配置，请检查 .env.1688.local。"}

    requested_queries = [normalized]
    for related in related_queries or []:
        candidate = " ".join(related.split()).strip()
        if candidate and candidate not in requested_queries:
            requested_queries.append(candidate)
    if max_items > 10 and related_queries is None:
        requested_queries.extend((normalized + " 现货货源", normalized + " 批发货源"))

    products = []
    seen_urls = set()
    returned_candidates = 0
    failed_queries = []
    queries_used = []
    filtered_count = 0
    duplicate_count = 0
    for current_query in requested_queries:
        if len(products) >= max_items:
            break
        queries_used.append(current_query)
        try:
            offers = await _search_once(current_query, appcode, app_id)
        except ValueError as exc:
            failed_queries.append({"query": current_query, "error": str(exc)})
            continue
        returned_candidates += len(offers)
        for item in offers:
            product = _normalize_offer(item)
            if product is None or product["minimum_order_quantity"] > max_minimum_order_quantity:
                filtered_count += 1
                continue
            if product["url"] in seen_urls:
                duplicate_count += 1
                continue
            seen_urls.add(product["url"])
            if len(products) < max_items:
                products.append(product)
    return {
        "backend": "aliyun_market_1688_deepsearch",
        "market": "1688",
        "price_type": "B2B 货源阶梯报价，不是淘宝零售价或市场均价",
        "data_quality_note": "搜索样本并非市场抽样，报价尚未逐一在商品页复核，运费与税费未计入。",
        "query": normalized,
        "queries_used": queries_used,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "returned_candidates": returned_candidates,
        "filtered_count": filtered_count,
        "duplicate_count": duplicate_count,
        "clean_product_count": len(products),
        "failed_queries": failed_queries,
        "products": products,
    }


search_1688_products_tool = tool("search_1688_products")(search_1688_products)
