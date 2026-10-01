"""Provider-free tests for one-shot Worker capability expansion."""

from types import SimpleNamespace
import unittest

from langchain_core.messages import HumanMessage

from middlewares import ToolsetRouterMiddleware
from retrieval_models import RerankResult
from toolset_router import ToolsetRouter
from toolsets import ToolsetRegistry, ToolsetSpec
from workers.code_runtime import CodeStepRuntime
from tools.capability_expansion import (
    CAPABILITY_EXPANSION_EXHAUSTED,
    CAPABILITY_EXPANSION_REQUESTED,
    CapabilityExpansionRequest,
    request_additional_capabilities,
)


def tool(name):
    return SimpleNamespace(name=name)


def spec(name, tool_name, *, allow_runtime_expansion=True):
    return ToolsetSpec(
        name=name,
        description=f"{name} capability",
        routing_profile=(
            f"# {name}\n\n## 选择它\n{name} work.\n\n"
            "## 不选择它\nOther work.\n\n## 典型命令\nDo the work."
        ),
        routing_threshold=0.35,
        instructions=f"Use {name} capability.",
        required_tool_names=(tool_name,),
        allow_runtime_expansion=allow_runtime_expansion,
    )


class Reranker:
    async def arerank(self, query, documents, top_k):
        return [
            RerankResult(index=index, text=text, score=1.0 - index * 0.1)
            for index, text in enumerate(documents)
        ][:top_k]


class Selector:
    def __init__(self, selections):
        self.selections = list(selections)
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class Bound:
            async def ainvoke(self, messages):
                outer.calls.append(messages)
                selected = outer.selections.pop(0)
                return {
                    "parsed": schema.model_validate({
                        "reason": "The required capability is grounded by the request.",
                        "selected_toolsets": selected,
                    }),
                    "parsing_error": None,
                }

        return Bound()


class Request:
    def __init__(self, tools, state, messages=None):
        self.tools = list(tools)
        self.state = dict(state)
        self.messages = list(messages or [HumanMessage(content="task")])
        self.model = SimpleNamespace()

    def override(self, **changes):
        return Request(
            changes.get("tools", self.tools),
            changes.get("state", self.state),
            changes.get("messages", self.messages),
        )


class CapabilityExpansionTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_only_browser_group_is_not_visible_to_runtime_expansion(self):
        registry = ToolsetRegistry([
            spec("ALPHA", "alpha_tool"),
            spec(
                "BROWSER_AUTOMATION",
                "browser_click",
                allow_runtime_expansion=False,
            ),
            spec("BETA", "beta_tool"),
        ])
        selector = Selector([["BROWSER_AUTOMATION"]])
        router = ToolsetRouter(Reranker(), registry=registry)

        metadata_names = {
            item["name"]
            for item in registry.build_router_metadata([
                tool("alpha_tool"), tool("browser_click"), tool("beta_tool")
            ])
        }
        self.assertIn("BROWSER_AUTOMATION", metadata_names)

        decision = await router.expand_with_model(
            selector,
            primary_task="continue the research",
            user_request_window="find public information",
            available_tools=[
                tool("alpha_tool"), tool("browser_click"), tool("beta_tool")
            ],
            mounted_toolsets=("ALPHA",),
            capability_request={"missing_capability": "more public evidence"},
        )

        self.assertIsNone(decision)
        self.assertNotIn("BROWSER_AUTOMATION", selector.calls[0][0]["content"])
        self.assertNotIn("BROWSER_AUTOMATION", selector.calls[0][1]["content"])

    async def test_expansion_selector_sees_full_catalog_but_can_mount_only_eligible_groups(self):
        registry = ToolsetRegistry([
            spec("ALPHA", "alpha_tool"),
            spec("BETA", "beta_tool"),
            spec("GAMMA", "gamma_tool"),
        ])
        selector = Selector([["BETA"]])
        router = ToolsetRouter(Reranker(), registry=registry)

        decision = await router.expand_with_model(
            selector,
            primary_task="continue with beta work",
            user_request_window="perform alpha then beta",
            available_tools=[tool("alpha_tool"), tool("beta_tool")],
            mounted_toolsets=("ALPHA",),
            capability_request={"missing_capability": "beta work"},
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.selected_toolset_names, ("BETA",))
        system_prompt = selector.calls[0][0]["content"]
        user_payload = selector.calls[0][1]["content"]
        self.assertIn("JSON object", system_prompt)
        self.assertIn('"name":"GAMMA"', system_prompt)
        self.assertIn('"available_to_current_worker":false', system_prompt)
        self.assertIn('"remaining_toolsets":["BETA"]', user_payload)

    async def test_first_request_keeps_mounted_group_and_adds_selected_group(self):
        registry = ToolsetRegistry([spec("ALPHA", "alpha_tool"), spec("BETA", "beta_tool")])
        selector = Selector([["ALPHA"], ["BETA"]])
        middleware = ToolsetRouterMiddleware(
            Reranker(),
            selector_model=selector,
            baseline_tool_names=("request_additional_capabilities", "report_general_result"),
        )
        middleware.toolset_router = ToolsetRouter(Reranker(), registry=registry)
        available = [
            tool("alpha_tool"), tool("beta_tool"),
            tool("request_additional_capabilities"), tool("report_general_result"),
        ]
        base_state = {
            "conversation_id": "expansion-1",
            "toolset_route_query": "perform alpha work then deliver beta",
            "toolset_route_full_user_request": "perform alpha work then deliver beta",
            "executor_tool_run_limit": 8,
        }
        observed = []

        async def handler(request):
            observed.append([item.name for item in request.tools])
            return "ok"

        await middleware.awrap_model_call(Request(available, base_state), handler)
        expanded_state = {
            **base_state,
            "capability_expansion_requests": [{
                "blocked_objective": "deliver the result",
                "missing_capability": "delivery",
                "why_current_capabilities_are_insufficient": "only transformation is mounted",
                "evidence_refs": ["S1"],
                "resolved_evidence": [{"ref": "S1", "content": "deliver the result"}],
            }],
        }
        await middleware.awrap_model_call(Request(available, expanded_state), handler)

        self.assertIn("alpha_tool", observed[0])
        self.assertNotIn("beta_tool", observed[0])
        self.assertIn("alpha_tool", observed[1])
        self.assertIn("beta_tool", observed[1])
        self.assertEqual(len(selector.calls), 2)
        self.assertIn('"mode":"ONE_SHOT_EXPANSION"', selector.calls[1][1]["content"])

    async def test_second_request_exposes_only_closeout_control(self):
        registry = ToolsetRegistry([spec("ALPHA", "alpha_tool")])
        selector = Selector([["ALPHA"]])
        middleware = ToolsetRouterMiddleware(
            Reranker(),
            selector_model=selector,
            baseline_tool_names=("request_additional_capabilities", "report_general_result"),
        )
        middleware.toolset_router = ToolsetRouter(Reranker(), registry=registry)
        available = [
            tool("alpha_tool"), tool("request_additional_capabilities"),
            tool("report_general_result"),
        ]
        state = {
            "conversation_id": "expansion-2",
            "toolset_route_query": "alpha work",
            "executor_tool_run_limit": 8,
        }

        async def handler(request):
            self.visible = [item.name for item in request.tools]
            return "ok"

        await middleware.awrap_model_call(Request(available, state), handler)
        await middleware.awrap_model_call(Request(available, {
            **state,
            "capability_expansion_requests": [
                {"status": "CAPABILITY_EXPANSION_REQUESTED"},
                {"status": "CAPABILITY_EXPANSION_EXHAUSTED"},
            ],
        }), handler)
        self.assertEqual(self.visible, ["report_general_result"])
        self.assertEqual(len(selector.calls), 1)


class ReviewerRoutingTests(unittest.TestCase):
    def test_code_reviewer_drops_worker_router_but_keeps_fixed_middleware(self):
        runtime = CodeStepRuntime.__new__(CodeStepRuntime)
        router = ToolsetRouterMiddleware(Reranker())
        fixed = object()
        runtime.middleware_factory = lambda: [fixed, router]
        self.assertEqual(runtime._reviewer_middleware(), [fixed])
        self.assertEqual(runtime._middleware(), [fixed, router])


class CapabilityExpansionToolTests(unittest.TestCase):
    def test_first_request_resolves_context_refs_and_second_forces_closeout(self):
        state = {
            "toolset_route_full_user_request": "把生成的文件发给我",
            "toolset_route_query": "交付已经生成的文件",
            "messages": [],
        }
        runtime = SimpleNamespace(state=state, tool_call_id="cap-1")
        request = CapabilityExpansionRequest(
            blocked_objective="向用户交付文件",
            missing_capability="向当前会话发送文件",
            why_current_capabilities_are_insufficient="当前只有文件生成能力",
            evidence_refs=["U1", "S1"],
        )
        first = request_additional_capabilities.func(request, runtime)
        update = first.update
        self.assertEqual(
            update["capability_expansion_requests"][0]["status"],
            CAPABILITY_EXPANSION_REQUESTED,
        )
        self.assertEqual(
            [item["ref"] for item in update["capability_expansion_requests"][0]["resolved_evidence"]],
            ["U1", "S1"],
        )

        runtime.state = {**state, **update}
        runtime.tool_call_id = "cap-2"
        second = request_additional_capabilities.func(request, runtime)
        self.assertTrue(second.update["worker_finalize_requested"])
        self.assertEqual(
            second.update["capability_expansion_requests"][-1]["status"],
            CAPABILITY_EXPANSION_EXHAUSTED,
        )
        self.assertIn("不要填写、推荐或猜测任何API名称", second.update["worker_finalize_reason"])


if __name__ == "__main__":
    unittest.main()
