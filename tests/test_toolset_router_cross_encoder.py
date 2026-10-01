from types import SimpleNamespace
import unittest

from retrieval_models import RerankResult
from toolset_router import ToolsetRouter, WebModelToolsetSelection
from toolsets import DEFAULT_TOOLSET_REGISTRY, NO_TOOL_ROUTE, ToolsetRegistry, ToolsetSpec


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


class FakeToolsetSelectionModel:
    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages):
        return {
            "parsed": {
                "reason": "PDF生成需要读文件并执行验证命令。",
                "selected_toolsets": ["SOFTWARE_DEVELOPMENT"],
            },
            "parsing_error": None,
        }


class CompositeToolsetSelectionModel:
    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages):
        return {
            "parsed": {
                "reason": "先查询官方来源，再把结论写入本地文件。",
                "selected_toolsets": ["WEB_RESEARCH", "FILE_EDITING"],
            },
            "parsing_error": None,
        }


class WebCheckpointSelectionModel:
    def __init__(self, checkpoint_value=True):
        self.checkpoint_value = checkpoint_value
        self.schemas = []
        self.messages = []

    def with_structured_output(self, schema, **kwargs):
        self.schemas.append(schema)
        return self

    async def ainvoke(self, messages):
        self.messages.append(messages)
        return {
            "parsed": {
                "reason": "当前Step需要沿论文引用继续核验。",
                "selected_toolsets": ["WEB_RESEARCH"],
                "require_progress_checkpoint": self.checkpoint_value,
            },
            "parsing_error": None,
        }


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

    async def test_hybrid_selector_can_resolve_composite_web_and_file_task(self):
        # hybrid_candidates does not include NO_TOOL, so registry order is
        # WEB_RESEARCH, BROWSER_AUTOMATION, FILE_EDITING.
        reranker = FakeReranker([0.92, 0.08, 0.87])
        router = ToolsetRouter(reranker, registry=self.registry)

        decision = await router.route_hybrid_with_model(
            CompositeToolsetSelectionModel(),
            primary_task="查询官方说明，然后把结论写进本地 Markdown",
            user_request_window="查询官方说明，然后把结论写进本地 Markdown",
            available_tools=self.tools,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(
            decision.selected_toolset_names,
            ("WEB_RESEARCH", "FILE_EDITING"),
        )
        self.assertEqual(decision.tool_names, ("web_search", "write_file"))

    async def test_web_selector_reuses_same_call_for_progress_checkpoint_decision(self):
        reranker = FakeReranker([0.92, 0.08, 0.20])
        router = ToolsetRouter(reranker, registry=self.registry)
        model = WebCheckpointSelectionModel(True)

        decision = await router.route_hybrid_with_model(
            model,
            primary_task="筛选论文后沿引用核对相互矛盾的实验结论",
            user_request_window="研究近五年的方法演进",
            available_tools=self.tools,
            select_progress_checkpoint=True,
        )

        self.assertIsNotNone(decision)
        self.assertTrue(decision.require_progress_checkpoint)
        self.assertIs(model.schemas[0], WebModelToolsetSelection)
        self.assertIn(
            "查北京故宫明天是否开放",
            model.messages[0][0]["content"],
        )

    def test_invalid_web_checkpoint_value_defaults_false_without_losing_toolsets(self):
        parsed = WebModelToolsetSelection.model_validate({
            "selected_toolsets": ["WEB_RESEARCH"],
            "require_progress_checkpoint": {"unexpected": "shape"},
        })

        self.assertEqual(parsed.selected_toolsets, ["WEB_RESEARCH"])
        self.assertFalse(parsed.require_progress_checkpoint)

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


class ToolsetRegistryCurrentNameTests(unittest.TestCase):
    def test_research_and_movie_groups_do_not_leak_browser_tools(self):
        research = DEFAULT_TOOLSET_REGISTRY.get("WEB_RESEARCH")
        movie = DEFAULT_TOOLSET_REGISTRY.get("MOVIE")
        browser = DEFAULT_TOOLSET_REGISTRY.get("BROWSER_AUTOMATION")

        self.assertIsNotNone(research)
        self.assertIsNotNone(movie)
        self.assertIsNotNone(browser)
        self.assertEqual(research.optional_tool_names, ("fetch_webpage",))
        self.assertEqual(movie.optional_tool_names, ("web_search",))
        self.assertFalse(browser.allow_runtime_expansion)
        self.assertFalse(any(
            name.startswith("browser_")
            for name in (*research.required_tool_names, *research.optional_tool_names)
        ))
        self.assertFalse(any(
            name.startswith("browser_")
            for name in (*movie.required_tool_names, *movie.optional_tool_names)
        ))

    def test_current_deep_agent_tool_names_resolve_without_compatibility_aliases(self):
        registry = ToolsetRegistry(
            [
                ToolsetSpec(
                    name="FILES",
                    description="file tools",
                    routing_profile="## 选择它\nfiles\n## 典型命令\nread files",
                    routing_threshold=0.35,
                    instructions="inspect files",
                    required_tool_names=("ls", "glob", "grep"),
                    optional_tool_names=("edit_file",),
                )
            ]
        )

        resolution = registry.resolve(
            "FILES",
            [_tool("ls"), _tool("glob"), _tool("grep"), _tool("edit_file")],
        )

        self.assertEqual(resolution.missing_required_tool_names, ())
        self.assertEqual(resolution.tool_names, ("ls", "glob", "grep", "edit_file"))

    def test_missing_tool_names_are_not_silently_aliased(self):
        registry = ToolsetRegistry(
            [
                ToolsetSpec(
                    name="LEGACY_FILES",
                    description="legacy file tools",
                    routing_profile="## 选择它\nlegacy files\n## 典型命令\nread files",
                    routing_threshold=0.35,
                    instructions="inspect files",
                    required_tool_names=("missing_test_tool",),
                )
            ]
        )
        resolution = registry.resolve("LEGACY_FILES", [_tool("ls")])
        self.assertEqual(resolution.missing_required_tool_names, ("missing_test_tool",))
        self.assertEqual(registry.build_router_metadata([_tool("ls")]), [])

    def test_pdf_code_step_exposes_current_file_and_execution_toolsets(self):
        tools = [
            _tool(name)
            for name in (
                "ls", "read_file", "write_file", "edit_file", "delete",
                "glob", "grep", "execute", "task",
            )
        ]
        names = {
            item["name"]
            for item in DEFAULT_TOOLSET_REGISTRY.build_router_metadata(tools)
        }
        self.assertEqual(
            names,
            {"FILE_INSPECTION", "FILE_EDITING", "SOFTWARE_DEVELOPMENT"},
        )
        decision = ToolsetRouter(
            FakeReranker(), registry=DEFAULT_TOOLSET_REGISTRY
        ).resolve_selected_toolsets(
            ["SOFTWARE_DEVELOPMENT"],
            tools,
            routing_task="读取交接菜谱，生成并验证 PDF。",
            allow_no_tool=False,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.selected_toolset_names, ("SOFTWARE_DEVELOPMENT",))
        self.assertEqual(
            decision.tool_names,
            ("read_file", "grep", "execute", "glob", "ls", "edit_file", "write_file"),
        )


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
