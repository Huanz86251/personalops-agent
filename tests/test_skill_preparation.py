"""Provider-free checks of routing, prompt isolation, recovery and budgets."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from agent import ask_worker
from hard_planning import run_hard_supervisor
from middlewares import DynamicExecutionBudgetMiddleware
from planning_models import PlanningContextPack
from prompt_loader import load_prompt, render_prompt, split_prompt
from skill_runtime import load_catalog, prepare_skills, prepare_skills_sync, skill_prompt
from skill_runtime.preparation import SkillChoice
from skill_runtime.middleware import RoleSkillsMiddleware
from workers.general_worker import create_general_worker
from planning_graph import build_planning_graph
from workers.registry import WorkerAgentRegistry
from workers.coordinator import WorkerGroupCoordinator
from eventing import AsyncEventStore
from evals.appworld.adapter import default_planning


@tool
def attachment_to_text(path: str) -> str:
    """Return a deterministic extracted document."""
    return "Page 1: probe"


@tool
def read_file(path: str) -> str:
    """Offline file boundary; no host files are accessed."""
    return "fixture"


@tool
def execute(command: str) -> str:
    """Offline execution boundary; no command is executed."""
    return "fixture"


class ProbeModel(BaseChatModel):
    selected: list[str] = Field(default_factory=lambda: ["general-read-documents"])
    selection_requests: list = Field(default_factory=list)
    execution_requests: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-skills-probe"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            if schema is SkillChoice:
                self.selection_requests.append(copy.deepcopy(messages))
                return {"parsed": {"skill_ids": self.selected, "reason": "SELECTOR_ONLY_MARKER"}, "parsing_error": None}
            self.execution_requests.append(copy.deepcopy(messages))
            return {"parsed": schema.model_validate({"action": "FINAL", "final_answer": "ready"}), "parsing_error": None}
        return RunnableLambda(respond)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.execution_requests.append(copy.deepcopy(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="executed"))])


class CatalogTests(unittest.TestCase):
    def test_expanded_catalog_is_reachable_only_in_declared_role_tool_scope(self):
        cases = [
            ("scheduler", [], ["plan-uncertainty"],
             {"code-debug-root-cause", "review-code-state"}),
            ("code", ["read_file", "execute"], ["code-debug-root-cause"],
             {"review-code-behavior", "review-code-state"}),
            ("reviewer", ["read_file", "execute"],
             ["review-code-behavior"],
             {"code-implement-core", "code-debug-root-cause"}),
            ("general", [], ["general-compare-options"],
             {"code-debug-root-cause", "general-read-documents"}),
        ]
        for role, tools, selected, excluded in cases:
            with self.subTest(role=role):
                model = ProbeModel(selected=selected)
                result = prepare_skills_sync(model, role=role, tools=tools,
                                             task="Scoped policy selection probe", mode="dynamic")
                self.assertEqual({s.name for s in result.selected}, set(selected))
                request = json.loads(model.selection_requests[0][0]["content"])
                candidates = request["available_skills"]
                self.assertTrue(all(set(item) == {"id", "description", "exclusive", "conflicts_with"}
                                    for item in candidates))
                self.assertFalse(excluded.intersection(item["id"] for item in candidates))
        for role, selected in [("code", ["code-debug-root-cause"]),
                               ("reviewer", ["review-code-state"])]:
            with self.subTest(role=role, unavailable_execute=True):
                model = ProbeModel(selected=selected)
                result = prepare_skills_sync(model, role=role, tools=["read_file"],
                                             task="No execution backend", mode="dynamic")
                self.assertEqual(result.selection_method, "no_candidates")
                self.assertEqual(model.selection_requests, [])

    def test_native_discovery_and_stable_content_hash(self):
        catalog = load_catalog()
        self.assertGreater(len(catalog), 3)
        self.assertEqual(catalog, load_catalog())
        self.assertTrue(all(s["content"].startswith("---") for s in catalog))

    def test_role_topic_and_actual_tool_filter(self):
        model = ProbeModel(selected=["general-read-documents"])
        snapshot = prepare_skills_sync(model, role="web", task="read a PDF", mode="dynamic",
                                      topics=["documents"], tools=["attachment_to_text"])
        self.assertEqual([s.name for s in snapshot.selected], ["general-read-documents"])
        blocked = prepare_skills_sync(model, role="web", task="read a PDF", mode="dynamic", topics=["documents"], tools=[])
        self.assertEqual(blocked.selection_method, "no_candidates")
        self.assertEqual(len(model.selection_requests), 1)

    def test_invalid_selection_is_not_injected_or_retried(self):
        model = ProbeModel(selected=["../../secrets"])
        result = prepare_skills_sync(model, role="scheduler", task="plan", mode="dynamic")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.selected, ())
        self.assertEqual(result.selection_method, "selection_error")
        self.assertEqual(len(model.selection_requests), 1)

    def test_off_fixed_and_budget_skip_make_no_model_call(self):
        model = ProbeModel()
        for mode, extra in [("off", {}), ("fixed", {"fixed_ids": ["plan-task-dependencies"]}),
                            ("dynamic", {"allow_model": False})]:
            result = prepare_skills_sync(model, role="scheduler", task="plan", mode=mode, **extra)
            self.assertEqual(result.model_calls, 0)
        self.assertEqual(model.selection_requests, [])
        greeting = prepare_skills_sync(model, role="scheduler", task={"user_request": "你好！"}, mode="dynamic")
        self.assertEqual(greeting.selection_method, "simple_request")
        self.assertEqual(greeting.model_calls, 0)

    def test_restore_uses_snapshot_without_reopening_catalog(self):
        model = ProbeModel(selected=["plan-task-dependencies"])
        frozen = prepare_skills_sync(model, role="scheduler", task="plan", mode="dynamic")
        with patch("skill_runtime.preparation.load_catalog", side_effect=AssertionError("must not reload")):
            restored = prepare_skills_sync(model, role="scheduler", task="continue", catalog=[], saved=frozen.model_dump(mode="json"))
        self.assertEqual(skill_prompt(frozen), skill_prompt(restored))
        self.assertEqual(len(model.selection_requests), 1)
        corrupted = frozen.model_dump(mode="json")
        corrupted["selected"][0]["content"] += "tampered"
        with self.assertRaises(ValueError):
            prepare_skills_sync(model, role="scheduler", task="resume", catalog=[], saved=corrupted)

    def test_static_prompt_prefix_does_not_contain_task_values(self):
        a = split_prompt(render_prompt("planning/supervisor", hard_context="TASK_ALPHA", max_steps_per_plan=3))
        b = split_prompt(render_prompt("planning/supervisor", hard_context="TASK_BETA", max_steps_per_plan=5))
        self.assertEqual(a[0], b[0])
        self.assertNotIn("TASK_ALPHA", a[0])
        self.assertEqual(a, b)  # 阶段协议不再内嵌任务；任务由消息流单独追加。
        self.assertIn("request_code_worker_repair", load_prompt("reviewers/code"))


class RolePreparationTests(unittest.IsolatedAsyncioTestCase):
    def test_replanned_worker_skill_selection_uses_compact_new_contract(self):
        middleware = RoleSkillsMiddleware(None, "general", ())
        replan_context = {
            "user_request": "导出后关闭账户",
            "accepted_steps": [{"step_id": 1, "status": "COMPLETED"}],
            "previous_failure": "CSV格式验收失败",
            "new_step_id": 3,
            "objective": "重新导出",
            "success_criteria": ["官方格式通过"],
        }
        options = middleware._options({
            "messages": [{"role": "user", "content": "旧Worker指令"}],
            "skill_reselection_context": replan_context,
            "executor_model_run_limit": 5,
        })
        self.assertEqual(options["task"]["task_contract"], replan_context)

    async def test_single_policy_body_reaches_only_its_role_and_survives_resume(self):
        # Exercise the shared middleware used by both real CODE factories.
        # Scripted choices verify plumbing, not a provider's selection quality.
        catalog = load_catalog()
        cases = [
            ("code", ["code-implement-core"],
             "Fix a parser regression without breaking valid records."),
            ("reviewer", ["review-code-behavior"],
             "Independently verify persistent event replay and recovery."),
        ]
        for role, selected, task in cases:
            with self.subTest(role=role):
                model = ProbeModel(selected=selected)
                worker = create_agent(model, tools=[read_file, execute],
                    system_prompt=f"Role: {role}",
                    middleware=[RoleSkillsMiddleware(model, role, ["read_file", "execute"])],
                    checkpointer=InMemorySaver())
                config = {"configurable": {"thread_id": f"policy-{role}"}}
                first = await worker.ainvoke({
                    "messages": [{"role": "user", "content": task}],
                    "skill_mode": "dynamic", "skill_catalog": catalog,
                    "executor_model_run_limit": 5,
                }, config=config)
                injected = model.execution_requests[0][0].content
                self.assertIn(skill_prompt(first["role_skill_snapshot"]), injected)
                self.assertNotIn("SELECTOR_ONLY_MARKER", str(model.execution_requests))
                for asset in catalog:
                    body = asset["content"].split("---", 2)[2].strip()
                    if asset["name"] in selected:
                        self.assertIn(body, injected)
                    else:
                        self.assertNotIn(body, injected)
                with patch("skill_runtime.preparation.load_catalog",
                           side_effect=AssertionError("must use frozen snapshot")):
                    second = await worker.ainvoke({"messages": [{
                        "role": "user", "content": "Continue with updated evidence."
                    }]}, config=config)
                self.assertEqual(first["role_skill_snapshot"], second["role_skill_snapshot"])
                self.assertEqual(len(model.selection_requests), 1)
                self.assertEqual(injected, model.execution_requests[-1][0].content)

    async def test_scheduler_checkpoint_resumes_after_selection_without_reselection(self):
        model = ProbeModel(selected=["plan-task-dependencies"])
        with TemporaryDirectory() as directory:
            store = AsyncEventStore(Path(directory) / "events.sqlite")
            graph = build_planning_graph(
                simple_model=model, hard_model=model,
                worker_registry=WorkerAgentRegistry.general_worker_first(object()),
                worker_group_coordinator=WorkerGroupCoordinator(store),
                planning=default_planning(), model_output_max_tokens=2048,
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "scheduler-skill-checkpoint"}}
            paused = await graph.ainvoke({"context": PlanningContextPack(
                current_time="now", user_request="plan a code change", skill_mode="dynamic")},
                config=config, interrupt_after=["prepare_scheduler"])
            self.assertEqual(len(model.selection_requests), 1)
            self.assertEqual(model.execution_requests, [])
            self.assertEqual(paused["model_rounds_used"], 1)
            with patch("planning_graph.load_catalog", side_effect=AssertionError("catalog already frozen")):
                finished = await graph.ainvoke(None, config=config)
            self.assertEqual(len(model.selection_requests), 1)
            self.assertEqual(finished["model_rounds_used"], 2)
            self.assertEqual(finished["final_answer"], "ready")

    async def test_worker_prepares_once_and_keeps_execution_prefix_on_resume(self):
        model = ProbeModel()
        worker = create_general_worker(model, tools=[attachment_to_text],
            middleware=[DynamicExecutionBudgetMiddleware()], checkpointer=InMemorySaver())
        common = {"skill_mode": "dynamic", "executor_model_run_limit": 5, "executor_tool_run_limit": 4,
                  "memory_context": "initial memory", "show_all_toolsets_run_limit": 0}
        first = await ask_worker(worker, "Read the supplied PDF.", thread_id="skills-worker", state_update=common, return_details=True)
        self.assertEqual(first["execution_summary"]["model_call_count"], 2)
        self.assertEqual(len(model.selection_requests), 1)
        first_prompt = str(model.execution_requests[0][0].content)
        self.assertIn("[general-read-documents]", first_prompt)
        self.assertNotIn("SELECTOR_ONLY_MARKER", first_prompt)
        self.assertNotIn("SELECTOR_ONLY_MARKER", str(first["current_turn_messages"]))
        second = await ask_worker(worker, "Continue with page 2.", thread_id="skills-worker",
            state_update={**common, "memory_context": "changed memory"}, return_details=True)
        self.assertEqual(len(model.selection_requests), 1)
        self.assertEqual(second["execution_summary"]["model_call_count"], 1)
        self.assertEqual(first_prompt, str(model.execution_requests[1][0].content))
        history = model.execution_requests[1]
        self.assertTrue(any("changed memory" in str(m.content) for m in history))
        self.assertEqual(first["role_skill_snapshot"], second["role_skill_snapshot"])

    async def test_scheduler_execution_uses_original_task_not_selector_transcript(self):
        model = ProbeModel(selected=["plan-task-dependencies"])
        snapshot = await prepare_skills(model, role="scheduler", task="ORIGINAL_USER_REQUEST", mode="dynamic")
        context = PlanningContextPack(current_time="now", user_request="ORIGINAL_USER_REQUEST",
            role_skill_snapshots={"scheduler": snapshot.model_dump(mode="json")})
        result = await run_hard_supervisor(model, context=context, max_steps_per_plan=3)
        self.assertFalse(result.used_fallback)
        messages = model.execution_requests[0]
        self.assertIn("[plan-task-dependencies]", messages[0]["content"])
        self.assertNotIn("ORIGINAL_USER_REQUEST", messages[0]["content"])
        self.assertIn("ORIGINAL_USER_REQUEST", messages[1]["content"])
        self.assertNotIn("SELECTOR_ONLY_MARKER", str(messages))

    async def test_low_worker_budget_keeps_execution_available(self):
        model = ProbeModel()
        worker = create_general_worker(model, tools=[attachment_to_text], middleware=[DynamicExecutionBudgetMiddleware()])
        result = await worker.ainvoke({"messages": [{"role":"user", "content":"read PDF"}],
            "skill_mode":"dynamic", "executor_model_run_limit":1, "executor_tool_run_limit":1, "show_all_toolsets_run_limit":0})
        self.assertEqual(model.selection_requests, [])
        self.assertEqual(result["executor_model_calls_used"], 1)
        self.assertEqual(result["role_skill_snapshot"]["selection_method"], "budget_skip")


if __name__ == "__main__":
    unittest.main()
