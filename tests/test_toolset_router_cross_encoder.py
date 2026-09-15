from types import SimpleNamespace
import unittest

from retrieval_models import RerankResult
from toolset_router import ToolsetRouter
from toolsets import NO_TOOL_ROUTE, ToolsetRegistry, ToolsetSpec


def _tool(name):
    return SimpleNamespace(name=name)


def _spec(name, tool_name, threshold=0.35, *, exclusive=False):
    return ToolsetSpec(
        name=name,
        description=f"{name} short description",
        routing_profile=(
            f"# {name} 能力卡\n\n"
            "## 选择它\nRelevant operation.\n\n"
            "## 不选择它\nMisleading negative keyword.\n\n"
            "## 典型命令\nDo the relevant operation.\n\n"
            "## 易混淆边界\nBoundary notes."
        ),
        routing_threshold=threshold,
        instructions=f"Use {tool_name} safely.",
        required_tool_names=(tool_name,),
        exclusive=exclusive,
    )


class FakeReranker:
    def __init__(self, scores=None, error=None):
        self.scores = list(scores or ())
        self.error = error
        self.calls = []

    async def arerank(self, query, documents, top_k):
        self.calls.append((query, list(documents), top_k))
        if self.error is not None:
            raise self.error
        results = [
            RerankResult(index=index, text=text, score=self.scores[index])
            for index, text in enumerate(documents)
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


class ToolsetCrossEncoderRouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = ToolsetRegistry(
            [
                _spec("WEB_RESEARCH", "web_search"),
                _spec("BROWSER_AUTOMATION", "browser_click"),
                _spec("FILE_EDITING", "write_file"),
            ]
        )
        self.tools = [_tool("web_search"), _tool("browser_click"), _tool("write_file")]

    async def test_scores_all_available_cards_in_one_batch_and_selects_relevant_group(self):
        # Document order is NO_TOOL followed by registry order.
        reranker = FakeReranker([0.08, 0.91, 0.12, 0.20])
        router = ToolsetRouter(reranker, registry=self.registry)

        decision = await router.route("查一下今天的新闻", self.tools)

        self.assertEqual(decision.selected_toolset_names, ("WEB_RESEARCH",))
        self.assertEqual(decision.tool_names, ("web_search",))
        self.assertEqual(len(reranker.calls), 1)
        self.assertEqual(reranker.calls[0][2], 4)
        self.assertFalse(
            any("Misleading negative keyword" in item for item in reranker.calls[0][1])
        )

    async def test_cross_encoder_selects_only_the_highest_relevant_group(self):
        reranker = FakeReranker([0.03, 0.80, 0.08, 0.88])
        router = ToolsetRouter(reranker, registry=self.registry, max_toolsets=3)

        decision = await router.route("查询官方说明并写入本地文件", self.tools)

        self.assertEqual(
            decision.selected_toolset_names,
            ("FILE_EDITING",),
        )

    async def test_exclusive_primary_group_suppresses_other_business_groups(self):
        registry = ToolsetRegistry(
            [
                _spec("APPWORLD", "appworld_execute", exclusive=True),
                _spec("WEB_RESEARCH", "web_search"),
                _spec("FILE_EDITING", "write_file"),
            ]
        )
        tools = [
            _tool("appworld_execute"),
            _tool("web_search"),
            _tool("write_file"),
        ]
        # NO_TOOL、APPWORLD、WEB_RESEARCH、FILE_EDITING。
        # WEB_RESEARCH分数足够成为副组，但APPWORLD是最高分独占组。
        router = ToolsetRouter(
            FakeReranker([0.02, 0.91, 0.89, 0.10]),
            registry=registry,
            max_toolsets=3,
        )

        decision = await router.route("在AppWorld中查询并操作模拟应用", tools)

        self.assertEqual(decision.selected_toolset_names, ("APPWORLD",))
        self.assertEqual(decision.tool_names, ("appworld_execute",))

    async def test_appworld_reviewer_variant_routes_discover_and_verify(self):
        registry = ToolsetRegistry(
            [
                ToolsetSpec(
                    name="APPWORLD",
                    description="AppWorld",
                    routing_profile="## 选择它\nAppWorld\n## 典型命令\nverify state",
                    routing_threshold=0.35,
                    instructions="discover then verify",
                    required_tool_names=("appworld_discover",),
                    optional_tool_names=("appworld_execute", "appworld_verify"),
                    exclusive=True,
                ),
                _spec("WEB_RESEARCH", "web_search"),
            ]
        )
        tools = [_tool("appworld_discover"), _tool("appworld_verify"), _tool("web_search")]
        # NO_TOOL、APPWORLD、WEB_RESEARCH。
        router = ToolsetRouter(
            FakeReranker([0.01, 0.93, 0.90]),
            registry=registry,
        )

        decision = await router.route("独立核验AppWorld状态", tools)

        self.assertEqual(decision.selected_toolset_names, ("APPWORLD",))
        self.assertEqual(decision.tool_names, ("appworld_discover", "appworld_verify"))

    async def test_returns_no_tool_only_with_absolute_score_and_margin(self):
        reranker = FakeReranker([0.82, 0.10, 0.06, 0.12])
        router = ToolsetRouter(reranker, registry=self.registry)

        decision = await router.route("解释一下什么是递归", self.tools)

        self.assertEqual(decision.selected_toolset_names, (NO_TOOL_ROUTE,))
        self.assertEqual(decision.tools, ())

    async def test_can_exclude_no_tool_before_cross_encoder_scoring(self):
        reranker = FakeReranker([0.91, 0.10, 0.06])
        router = ToolsetRouter(reranker, registry=self.registry)

        decision = await router.route(
            "查一下今天的新闻",
            self.tools,
            allow_no_tool=False,
        )

        self.assertEqual(decision.selected_toolset_names, ("WEB_RESEARCH",))
        scored_documents = reranker.calls[0][1]
        self.assertEqual(len(scored_documents), 3)
        self.assertFalse(any("NO_TOOL" in item for item in scored_documents))

    async def test_uncertain_scores_fail_open_instead_of_hiding_tools(self):
        reranker = FakeReranker([0.40, 0.34, 0.12, 0.22])
        router = ToolsetRouter(reranker, registry=self.registry)

        decision = await router.route("继续按刚才说的做", self.tools)

        self.assertIsNone(decision)

    async def test_unavailable_group_is_not_scored_or_selected(self):
        reranker = FakeReranker([0.05, 0.92, 0.99])
        router = ToolsetRouter(reranker, registry=self.registry)

        decision = await router.route(
            "搜索后编辑文件",
            [_tool("web_search"), _tool("write_file")],
        )

        self.assertEqual(decision.selected_toolset_names, ("FILE_EDITING",))
        scored_documents = reranker.calls[0][1]
        self.assertFalse(any("BROWSER_AUTOMATION" in item for item in scored_documents))

    async def test_model_failure_requests_conservative_fallback(self):
        router = ToolsetRouter(
            FakeReranker(error=RuntimeError("offline")),
            registry=self.registry,
        )

        self.assertIsNone(await router.route("查新闻", self.tools))


class ToolsetRegistryAliasTests(unittest.TestCase):
    def test_deep_agent_tool_aliases_resolve_without_duplicate_tools(self):
        registry = ToolsetRegistry(
            [
                ToolsetSpec(
                    name="FILES",
                    description="file tools",
                    routing_profile="## 选择它\nfiles\n## 典型命令\nread files",
                    routing_threshold=0.35,
                    instructions="inspect files",
                    required_tool_names=("list_directory", "find_files", "grep_files"),
                    optional_tool_names=("replace_in_file",),
                )
            ]
        )

        resolution = registry.resolve(
            "FILES",
            [_tool("ls"), _tool("glob"), _tool("grep"), _tool("edit_file")],
        )

        self.assertEqual(resolution.missing_required_tool_names, ())
        self.assertEqual(resolution.tool_names, ("ls", "glob", "grep", "edit_file"))


class ToolsetRoutingCardTests(unittest.TestCase):
    def test_spec_requires_a_profile_and_valid_threshold(self):
        with self.assertRaisesRegex(ValueError, "routing_profile"):
            ToolsetRegistry([
                ToolsetSpec(
                    name="EMPTY_PROFILE",
                    description="description",
                    routing_profile="",
                    routing_threshold=0.35,
                    instructions="instructions",
                    required_tool_names=("tool",),
                )
            ])

        with self.assertRaisesRegex(ValueError, "routing_threshold"):
            ToolsetRegistry([
                ToolsetSpec(
                    name="BAD_THRESHOLD",
                    description="description",
                    routing_profile="## 选择它\npositive\n## 典型命令\nexample",
                    routing_threshold=1.1,
                    instructions="instructions",
                    required_tool_names=("tool",),
                )
            ])


if __name__ == "__main__":
    unittest.main()
