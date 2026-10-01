"""Offline contract checks for the 1688 Marketplace search adapter."""

import unittest
from unittest.mock import AsyncMock, patch

from tools.market_1688 import _final_offers, _normalize_offer, search_1688_products
from toolsets import DEFAULT_TOOLSET_REGISTRY
from langchain_core.tools import StructuredTool


def offer(item_id: int, title: str, price: float = 51, quantity: int = 1):
    return {
        "offerInfo": {
            "title": title,
            "pcDetailUrl": f"https://detail.1688.com/offer/{item_id}.html",
            "minPrice": "49",
            "offerPriceRanges": [
                {"price": price, "beginAmount": quantity},
                {"price": 49, "beginAmount": 1000},
            ],
            "sellerInfo": {"companyName": "Example supplier"},
            "__track__": "private-tracking-metadata",
        }
    }


class Market1688Tests(unittest.IsolatedAsyncioTestCase):
    def test_reads_nested_completed_event(self):
        event = {"status": "completed", "output": [{"content": [{"data": {
            "responseData": {"status": "returning", "data": [offer(1, "女士连衣裙")]}
        }}]}]}
        self.assertEqual(len(_final_offers(event)), 1)

    def test_only_compact_valid_product_fields_reach_model(self):
        product = _normalize_offer(offer(1, "女士连衣裙"))
        self.assertEqual(product["entry_unit_price_cny"], 51)
        self.assertEqual(product["price_tiers"][1]["unit_price_cny"], 49)
        self.assertNotIn("__track__", str(product))
        self.assertNotIn("minPrice", str(product))
        higher_minimum = offer(5, "女士连衣裙")
        higher_minimum["offerInfo"]["minPurchaseCnt"] = "1000"
        self.assertEqual(_normalize_offer(higher_minimum)["entry_unit_price_cny"], 49)
        self.assertEqual(_normalize_offer(higher_minimum)["minimum_order_quantity"], 1000)
        self.assertIsNone(_normalize_offer(offer(2, "邮费补差链接", 1)))
        self.assertIsNone(_normalize_offer(offer(3, "补邮费差价专用链接", 1)))
        self.assertIsNone(_normalize_offer(offer(4, "1元链接 补拍运费", 1)))

    async def test_three_queries_merge_clean_unique_results_up_to_thirty(self):
        batches = [
            [offer(i, f"女士连衣裙 {i}") for i in range(1, 11)],
            [offer(1, "女士连衣裙 1"), offer(11, "邮费补差链接")]
            + [offer(i, f"女士连衣裙 {i}") for i in range(12, 21)],
            [offer(i, f"女士连衣裙 {i}") for i in range(21, 31)],
        ]
        with (patch("tools.market_1688._credentials", return_value=("test-code", "test-app")),
              patch("tools.market_1688._search_once", new=AsyncMock(side_effect=batches)) as search):
            result = await search_1688_products("女士连衣裙", max_items=30)
        self.assertEqual(search.await_count, 3)
        self.assertEqual(result["returned_candidates"], 31)
        self.assertEqual(result["clean_product_count"], 29)
        self.assertEqual(len(result["products"]), 29)
        self.assertEqual(len({p["url"] for p in result["products"]}), 29)
        self.assertEqual(result["price_type"], "B2B 货源阶梯报价，不是淘宝零售价或市场均价")

    async def test_large_minimum_order_is_filtered_by_default(self):
        large_lot = offer(42, "女士连衣裙")
        large_lot["offerInfo"]["minPurchaseCnt"] = "200"
        with (patch("tools.market_1688._credentials", return_value=("test-code", "test-app")),
              patch("tools.market_1688._search_once", new=AsyncMock(return_value=[large_lot]))):
            result = await search_1688_products("女士连衣裙", max_items=1)
        self.assertEqual(result["clean_product_count"], 0)
        self.assertEqual(result["filtered_count"], 1)

    def test_group_hidden_without_real_tool(self):
        self.assertNotIn("ECOMMERCE_1688", {
            entry["name"] for entry in DEFAULT_TOOLSET_REGISTRY.build_router_metadata([])
        })
        tool = StructuredTool.from_function(
            name="search_1688_products", description="fixture", func=lambda: None,
        )
        self.assertIn("ECOMMERCE_1688", {
            entry["name"] for entry in DEFAULT_TOOLSET_REGISTRY.build_router_metadata([tool])
        })
        self.assertFalse(DEFAULT_TOOLSET_REGISTRY.get("ECOMMERCE_1688").exclusive)


if __name__ == "__main__":
    unittest.main()
