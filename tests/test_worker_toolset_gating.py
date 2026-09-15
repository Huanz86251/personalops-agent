"""Provider-free tests for Worker toolset routing and final RAG gating."""

import unittest
from types import SimpleNamespace

from knowledge_rag.middleware import KnowledgeMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from middlewares import ToolsetRouterMiddleware
from retrieval_models import RerankResult


def _tool(name):
    return SimpleNamespace(name=name)


class ContentAwareReranker:
    def __init__(self):
        self.queries = []

    async def arerank(self, query, documents, top_k):
        self.queries.append(query)
        results = []
        for index, document in enumerate(documents):
            if "能力组：APPWORLD" in document:
                score = 0.92
            elif "能力组：WEB_RESEARCH" in document:
                score = 0.90
            elif "无需调用任何工具" in document:
                score = 0.02
            else:
                score = 0.05
            results.append(RerankResult(index=index, text=document, score=score))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


class UncertainReranker:
    def __init__(self):
        self.queries = []

    async def arerank(self, query, documents, top_k):
        self.queries.append(query)
        return [
            RerankResult(index=index, text=document, score=0.10)
            for index, document in enumerate(documents)
        ][:top_k]


class FakeRequest:
    def __init__(self, *, tools, state, messages=None, model=None):
        self.tools = list(tools)
        self.state = dict(state)
        self.messages = list(messages or [HumanMessage(content="这段用户文本不用于工具选择")])
        self.model = model

    def override(self, **changes):
        return FakeRequest(
            tools=changes.get("tools", self.tools),
            state=changes.get("state", self.state),
            messages=changes.get("messages", self.messages),
            model=changes.get("model", self.model),
        )


class SequencedReranker:
    def __init__(self, score_sequences):
        self.score_sequences = [list(items) for items in score_sequences]
        self.queries = []

    async def arerank(self, query, documents, top_k):
        self.queries.append(query)
        scores = self.score_sequences[len(self.queries) - 1]
        results = [
            RerankResult(index=index, text=document, score=scores[index])
            for index, document in enumerate(documents)
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


class StructuredToolsetModel:
    def __init__(self, selected_toolsets):
        self.selected_toolsets = selected_toolsets
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class Bound:
            async def ainvoke(self, messages):
                outer.calls.append(messages)
                return {
                    "parsed": schema.model_validate({
                        "reason": "当前Step需要该能力组。",
                        "selected_toolsets": outer.selected_toolsets,
                    }),
                    "raw": None,
                    "parsing_error": None,
                }

        return Bound()


class WorkerToolsetGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_disabled_tools_skips_router_and_hides_business_and_show_all(self):
        reranker = UncertainReranker()
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=("report_general_result",),
        )
        captured = []

        async def handler(selected_request):
            captured.extend(tool.name for tool in selected_request.tools)
            return "ok"

        await middleware.awrap_model_call(
            FakeRequest(
                tools=[
                    _tool("appworld_discover"),
                    _tool("show_all_toolsets"),
                    _tool("report_general_result"),
                ],
                state={
                    "conversation_id": "conv-tools-disabled",
                    "toolset_route_query": "解释已提供的句子",
                    "step_tool_access": "DISABLED",
                    "executor_tool_run_limit": 20,
                },
            ),
            handler,
        )

        self.assertEqual(reranker.queries, [])
        self.assertEqual(captured, ["report_general_result"])

    async def test_scheduler_enabled_tools_excludes_no_tool_from_reranker_candidates(self):
        reranker = SequencedReranker([[0.91]])
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=("report_general_result",),
        )
        captured = []

        async def handler(selected_request):
            captured.extend(tool.name for tool in selected_request.tools)
            return "ok"

        await middleware.awrap_model_call(
            FakeRequest(
                tools=[_tool("appworld_discover"), _tool("report_general_result")],
                state={
                    "conversation_id": "conv-tools-enabled",
                    "toolset_route_query": "读取模拟应用数据",
                    "step_tool_access": "ENABLED",
                    "executor_tool_run_limit": 20,
                },
            ),
            handler,
        )

        self.assertEqual(reranker.queries, ["读取模拟应用数据"])
        self.assertIn("appworld_discover", captured)
        self.assertIn("report_general_result", captured)

    async def test_uncertain_route_is_cached_without_exposing_all_tools(self):
        reranker = UncertainReranker()
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=("report_general_result",),
        )
        state = {
            "conversation_id": "conv-uncertain-cache",
            "toolset_route_query": "读取消息并发送回复",
            "executor_tool_run_limit": 20,
        }
        tools = [
            _tool("appworld_discover"),
            _tool("appworld_execute"),
            _tool("report_general_result"),
        ]
        captured = []

        async def handler(selected_request):
            captured.append([tool.name for tool in selected_request.tools])
            return "ok"

        await middleware.awrap_model_call(FakeRequest(tools=tools, state=state), handler)
        await middleware.awrap_model_call(FakeRequest(tools=tools, state=state), handler)

        self.assertEqual(reranker.queries, ["读取消息并发送回复"])
        self.assertEqual(captured, [["report_general_result"]] * 2)


    async def test_legacy_show_all_result_does_not_reenable_all_tools(self):
        reranker = UncertainReranker()
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=("report_general_result",),
        )
        tools = [
            _tool("appworld_discover"),
            _tool("web_search"),
            _tool("show_all_toolsets"),
            _tool("report_general_result"),
        ]
        request = FakeRequest(
            tools=tools,
            state={
                "conversation_id": "conv-show-all",
                "toolset_route_query": "当前Step",
                "executor_tool_run_limit": 20,
            },
            messages=[
                HumanMessage(content="当前Step"),
                ToolMessage(
                    content="SHOW_ALL_TOOLSETS: 当前可见工具无法完成下一步",
                    tool_call_id="show-all-1",
                    name="show_all_toolsets",
                ),
            ],
        )
        captured = []

        async def handler(selected_request):
            captured.extend(tool.name for tool in selected_request.tools)
            return "ok"

        await middleware.awrap_model_call(request, handler)

        self.assertEqual(captured, ["report_general_result"])
        self.assertEqual(reranker.queries, ["当前Step"])

    async def test_weighted_route_combines_step_and_user_head_tail_scores(self):
        reranker = SequencedReranker([
            [0.10],
            [0.92],
        ])
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=("report_general_result",),
        )
        state = {
            "conversation_id": "conv-second-pass",
            "toolset_route_query": "当前Step短任务",
            "toolset_route_fallback_query": "用户原话开头 " + "x" * 2000 + " 用户原话结尾",
            "executor_tool_run_limit": 20,
        }
        tools = [_tool("appworld_discover"), _tool("report_general_result")]
        captured = []

        async def handler(selected_request):
            captured.extend(tool.name for tool in selected_request.tools)
            return "ok"

        await middleware.awrap_model_call(FakeRequest(tools=tools, state=state), handler)

        self.assertEqual(len(reranker.queries), 2)
        self.assertEqual(reranker.queries[0], "当前Step短任务")
        self.assertIn("用户原话开头", reranker.queries[1])
        self.assertIn("用户原话结尾", reranker.queries[1])
        self.assertIn("appworld_discover", captured)

    async def test_two_low_scores_reuse_same_worker_model_for_short_selection(self):
        reranker = SequencedReranker([
            [0.10, 0.20],
            [0.11, 0.21],
        ])
        model = StructuredToolsetModel(["APPWORLD"])
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=("report_general_result",),
        )
        state = {
            "conversation_id": "conv-model-fallback",
            "toolset_route_query": "处理模拟应用记录",
            "toolset_route_fallback_query": "用户要求处理模拟应用中的记录",
            "toolset_route_full_user_request": "完整开头" + "中间信息" * 800 + "完整结尾",
            "executor_tool_run_limit": 20,
        }
        tools = [_tool("appworld_discover"), _tool("report_general_result")]
        captured = []

        async def handler(selected_request):
            captured.extend(tool.name for tool in selected_request.tools)
            return "ok"

        await middleware.awrap_model_call(
            FakeRequest(tools=tools, state=state, model=model), handler
        )

        self.assertEqual(len(model.calls), 1)
        fallback_payload = model.calls[0][1]["content"]
        self.assertNotIn('"name":"NO_TOOL"', fallback_payload)
        self.assertIn("完整开头", fallback_payload)
        self.assertIn("完整结尾", fallback_payload)
        self.assertGreater(len(fallback_payload), 3200)
        self.assertIn("appworld_discover", captured)
        self.assertIn("report_general_result", captured)

    async def test_explicit_step_query_reuses_route_across_runtime_messages(self):
        reranker = ContentAwareReranker()
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=("report_general_result",),
        )
        state = {
            "conversation_id": "conv-cache",
            "toolset_route_query": "在隔离AppWorld里完成模拟应用任务",
            "executor_tool_run_limit": 20,
        }
        tools = [
            _tool("appworld_discover"),
            _tool("appworld_execute"),
            _tool("report_general_result"),
        ]

        async def handler(_selected_request):
            return "ok"

        await middleware.awrap_model_call(
            FakeRequest(tools=tools, state=state),
            handler,
        )
        await middleware.awrap_model_call(
            FakeRequest(
                tools=tools,
                state=state,
                messages=[
                    HumanMessage(content="原始任务"),
                    HumanMessage(content="运行时追加的工具结果说明"),
                ],
            ),
            handler,
        )

        self.assertEqual(
            reranker.queries,
            ["在隔离AppWorld里完成模拟应用任务"],
        )

    async def test_appworld_is_exclusive_but_role_baselines_survive(self):
        reranker = ContentAwareReranker()
        middleware = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=(
                "search_knowledge",
                "read_knowledge",
                "report_general_result",
            ),
        )
        request = FakeRequest(
            tools=[
                _tool("appworld_discover"),
                _tool("appworld_execute"),
                _tool("web_search"),
                _tool("search_knowledge"),
                _tool("read_knowledge"),
                _tool("report_general_result"),
            ],
            state={
                "conversation_id": "conv-1",
                "toolset_route_query": "在隔离AppWorld里完成模拟应用任务",
                "executor_tool_run_limit": 20,
            },
        )
        captured = {}

        async def handler(selected_request):
            captured["tools"] = [tool.name for tool in selected_request.tools]
            return "ok"

        result = await middleware.awrap_model_call(request, handler)

        self.assertEqual(result, "ok")
        self.assertEqual(reranker.queries, ["在隔离AppWorld里完成模拟应用任务"])
        self.assertIn("appworld_discover", captured["tools"])
        self.assertIn("appworld_execute", captured["tools"])
        self.assertNotIn("web_search", captured["tools"])
        self.assertIn("search_knowledge", captured["tools"])
        self.assertIn("read_knowledge", captured["tools"])
        self.assertIn("report_general_result", captured["tools"])

    async def test_no_rag_grant_removes_public_rag_tools_after_routing(self):
        reranker = ContentAwareReranker()
        router = ToolsetRouterMiddleware(
            reranker,
            baseline_tool_names=(
                "search_knowledge",
                "read_knowledge",
                "report_general_result",
            ),
        )
        rag_gate = KnowledgeMiddleware("General Agent")
        request = FakeRequest(
            tools=[
                _tool("appworld_discover"),
                _tool("appworld_execute"),
                _tool("search_knowledge"),
                _tool("read_knowledge"),
                _tool("report_general_result"),
            ],
            state={
                "conversation_id": "conv-2",
                "toolset_route_query": "在隔离AppWorld里完成模拟应用任务",
                "executor_tool_run_limit": 20,
            },
        )
        captured = {}

        async def after_router(selected_request):
            final_request = rag_gate.filtered(selected_request)
            captured["tools"] = [tool.name for tool in final_request.tools]
            return "ok"

        await router.awrap_model_call(request, after_router)

        self.assertIn("appworld_execute", captured["tools"])
        self.assertIn("report_general_result", captured["tools"])
        self.assertNotIn("search_knowledge", captured["tools"])
        self.assertNotIn("read_knowledge", captured["tools"])


if __name__ == "__main__":
    unittest.main()
