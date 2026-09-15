"""Provider-free acceptance for the General self-report path and stable context."""

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from deepagents.backends.utils import create_file_data
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
from test_general_worker import business_probe
from test_planning_handoff_publication import planning_settings

from eventing import AsyncEventStore
from middlewares import DynamicExecutionBudgetMiddleware
from planning_graph import build_planning_graph
from planning_models import PlanningContextPack
from workers import WorkerAgentRegistry, WorkerGroupCoordinator
from workers.general_completion import GeneralResult, report_general_result
from workers.general_runtime import GeneralStepRuntime
from workers.general_worker import create_general_worker


class GeneralModel(BaseChatModel):
    status: str = "COMPLETED"
    write_file: bool = False
    natural_exit: bool = False
    requests: list[list[str]] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        if schema.__name__ == 'SkillChoice':
            return RunnableLambda(lambda messages: {'parsed': {'skill_ids': [], 'reason': 'Offline business probe needs no skill'}})
        return super().with_structured_output(schema, **kwargs)

    @property
    def _llm_type(self):
        return "offline-general-self-report"

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append([str(m.content) for m in messages])
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        observed = [m for m in messages[start:] if isinstance(m, ToolMessage)]
        has_criteria = any("HARNESS_CRITERIA:" in str(m.content) for m in messages)
        if self.tool_names == ["report_general_result"]:
            call = {"id": "finish", "name": "report_general_result", "args": {"result": {
                "status": "BLOCKED", "summary": "Execution stopped at budget boundary; no further work performed.",
                "unresolved_items": ["Need Scheduler help to complete remaining work."]}}}
        elif not observed:
            call = {
                "id": "probe",
                "name": "business_probe",
                "args": {"value": "observed result"},
            }
        elif self.natural_exit:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="Observed result; incomplete summary."
                        )
                    )
                ]
            )
        else:
            result = {
                "status": self.status,
                "summary": "Observed result, self checked.",
                "handoff_knowledge": [{"topic":"probe", "source":"business_probe", "usage":"Call business_probe() with no arguments.", "observed_result":"Returns the fixture observation.", "next_action":"Use the observed result without rerunning the probe."}],
                "evidence_tool_call_ids": ["probe"],
                "unresolved_items": ["Required tool is unavailable."]
                if self.status == "BLOCKED"
                else [],
            }
            if has_criteria:
                result["criterion_claims"] = [{
                    "criterion_id": "C1", "criterion": "Report the result",
                    "conclusion": "Observed result", "evidence_tool_call_ids": ["probe"],
                }]
            if self.write_file:
                result["files"] = [
                    {
                        "path": "/artifacts/note.md",
                        "description": "Reusable note",
                        "output_id": "note",
                    }
                ]
            call = {
                "id": "finish",
                "name": "report_general_result",
                "args": {"result": result},
            }
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="", tool_calls=[call]))
            ]
        )


class RepairingGeneralModel(GeneralModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append([str(m.content) for m in messages])
        rejected = any(
            "REPORT_SCHEMA_REJECTED" in str(getattr(message, "content", ""))
            for message in messages
        )
        claims = []
        if rejected:
            claims = [{
                "criterion_id": "C1",
                "criterion": "Report the result",
                "conclusion": "No business action was attempted before forced finalization.",
                "evidence_tool_call_ids": [],
            }]
        call = {
            "id": "repair-finish" if rejected else "invalid-finish",
            "name": "report_general_result",
            "args": {"result": {
                "status": "BLOCKED",
                "summary": "No business action was attempted before forced finalization.",
                "criterion_claims": claims,
                "unresolved_items": ["Business execution budget was unavailable."],
            }},
        }
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
        )


class SchedulerModel:
    def __init__(self, artifact=False, approve=True):
        self.schemas = []
        self.artifact = artifact
        self.approve = approve

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            if schema.__name__ == 'SkillChoice':
                return {'parsed': {'skill_ids': [], 'reason': 'Offline test has no matching skill'}}
            self.schemas.append(schema.__name__)
            if schema.__name__ == "StepReport":
                prompt = "\n".join(str(m.get("content", "") if isinstance(m, dict) else m.content) for m in messages)
                match = re.search(r'"review_ref":\s*"([^"]+)"', prompt)
                assert match or not self.artifact, "File review needs candidate reference"
                value = {"step_id": 1, "status": "COMPLETED", "summary": "Fixture note independently reviewed",
                         "stop_reason": "File review completed", "approved_artifact_refs": [match.group(1)] if self.approve and match else [],
                         "criterion_results": [{"criterion_id": "C1", "status": "MET", "evidence": [match.group(1)] if match else []}]}
                if not self.artifact:
                    value["status"] = "FAILED" if "HARNESS_FALLBACK" in prompt and "incomplete summary" not in prompt else "BLOCKED" if "Required tool is unavailable" in prompt else "PARTIAL" if "incomplete summary" in prompt else "COMPLETED"
                    if value["status"] != "COMPLETED":
                        value["criterion_results"][0]["status"] = "UNKNOWN"
            elif schema.__name__ == "SupervisorDecision":
                step = {
                    "step_id": 1,
                    "worker_kind": "GENERAL",
                    "objective": "Read and report the observation",
                    "success_criteria": ["Report the result"],
                }
                if self.artifact:
                    step["artifact_outputs"] = [
                        {
                            "output_id": "note",
                            "description": "Reusable note",
                            "disposition": "INTERNAL_HANDOFF",
                            "required": True,
                        }
                    ]
                value = {
                    "action": "PLAN",
                    "plan_objective": "Get the observation",
                    "plan_success_criteria": ["Report the result"],
                    "steps": [step],
                }
            else:
                value = {
                    "action": "FINAL",
                    "status": "COMPLETED",
                    "final_answer": "Main model summary.",
                    "unmet_success_criteria": [],
                }
            return {
                "parsed": schema.model_validate(value),
                "raw": AIMessage(content=json.dumps(value)),
                "parsing_error": None,
            }

        return RunnableLambda(respond)


class GeneralSimpleTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_rounds_no_reviewer_no_subagent_and_stable_context_prefix(self):
        model = GeneralModel()
        graph = create_general_worker(model, tools=[business_probe])
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Original PDF task unique marker")],
                "worker_id": "general",
                "event_id": "evt",
                "step_id": "1",
            }
        )
        self.assertEqual(result["general_result"]["status"], "COMPLETED")
        self.assertFalse(result.get("worker_review_requested"))
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(model.requests[1][: len(model.requests[0])], model.requests[0])
        for request in model.requests:
            self.assertEqual(
                "\n".join(request).count("Original PDF task unique marker"), 1
            )
        self.assertTrue(
            {
                "submit_for_review",
                "publish_worker_progress",
                "task",
                "execute",
            }.isdisjoint(model.tool_names)
        )

    async def test_new_turn_can_run_after_previous_self_report(self):
        model = GeneralModel()
        graph = create_general_worker(
            model, tools=[business_probe], checkpointer=InMemorySaver()
        )
        config = {"configurable": {"thread_id": "two-turns"}}
        for question in ["First task", "Second task"]:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=question)]}, config=config
            )
            self.assertEqual(result["general_result"]["status"], "COMPLETED")
        self.assertEqual(len(model.requests), 4)

    async def test_last_budget_round_is_reserved_for_real_summary(self):
        model = GeneralModel()
        graph = create_general_worker(
            model,
            tools=[business_probe],
            middleware=[DynamicExecutionBudgetMiddleware()],
        )
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="A task")],
                "executor_model_run_limit": 1,
                "executor_tool_run_limit": 3,
                "show_all_toolsets_run_limit": 0,
            }
        )
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(result["general_result"]["status"], "BLOCKED")
        self.assertTrue(result["general_result"]["forced_finalization"])
        self.assertEqual(result["executor_model_calls_used"], 1)
        self.assertEqual(result["worker_finalization_model_calls_used"], 1)
        self.assertEqual(model.tool_names, ["report_general_result"])



    async def test_invalid_final_report_is_repaired_by_same_worker_after_budget_limit(self):
        model = RepairingGeneralModel()
        graph = create_general_worker(
            model,
            tools=[business_probe],
            schema_repair_max_rounds=3,
            middleware=[DynamicExecutionBudgetMiddleware()],
        )
        result = await graph.ainvoke({
            "messages": [
                HumanMessage(
                    content='A task\nHARNESS_CRITERIA: {"C1":"Report the result"}'
                )
            ],
            "executor_model_run_limit": 1,
            "executor_tool_run_limit": 1,
            "show_all_toolsets_run_limit": 0,
        })

        self.assertEqual(len(model.requests), 2)
        self.assertEqual(result["general_result"]["status"], "BLOCKED")
        self.assertEqual(
            result["general_result"]["criterion_claims"][0]["criterion_id"],
            "C1",
        )
        self.assertEqual(result["executor_model_calls_used"], 1)
        self.assertEqual(result["worker_schema_repair_model_calls_used"], 1)
        self.assertEqual(result["worker_finalization_model_calls_used"], 2)

    async def test_tool_exhaustion_still_reports_without_more_business_actions(self):
        model = GeneralModel()
        graph = create_general_worker(model, tools=[business_probe], middleware=[DynamicExecutionBudgetMiddleware()])
        result = await graph.ainvoke({"messages": [HumanMessage(content="A task")],
            "executor_model_run_limit": 5, "executor_tool_run_limit": 1, "show_all_toolsets_run_limit": 0})
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(result["executor_model_calls_used"], 2)
        self.assertEqual(result["executor_tool_calls_used"], 1)
        self.assertEqual(result["general_result"]["status"], "BLOCKED")
        self.assertTrue(result["general_result"]["forced_finalization"])
        self.assertEqual(model.requests[1][:len(model.requests[0])], model.requests[0])

    async def test_zero_model_budget_cannot_spend_a_summary_call(self):
        model = GeneralModel()
        graph = create_general_worker(model, tools=[business_probe], middleware=[DynamicExecutionBudgetMiddleware()])
        result = await graph.ainvoke({"messages": [HumanMessage(content="A task")],
            "executor_model_run_limit": 0, "executor_tool_run_limit": 1, "show_all_toolsets_run_limit": 0})
        self.assertEqual(len(model.requests), 0)
        self.assertFalse(result.get("general_result"))

    def test_failed_report_preserves_error_not_last_tool_output(self):
        from types import SimpleNamespace
        from planning_graph import _build_general_step_report
        report = _build_general_step_report(SimpleNamespace(step_id=1), {
            "finish_reason": "BUDGET_EXHAUSTED", "stop_reason": "model limit",
            "final_answer": '{"irrelevant_last_document":true}',
            "messages": [ToolMessage(content="NameError: apis unavailable", tool_call_id="bad", status="error")],
        }, SimpleNamespace(attempts=[]))
        self.assertEqual(report.status, "FAILED")
        self.assertNotIn("irrelevant_last_document", report.summary)
        self.assertIn("NameError", report.errors[0])
        self.assertEqual(report.stop_reason, "model limit")

    def test_summary_does_not_repeat_even_when_business_budget_remains(self):
        from workers.general_completion import GeneralBudgetMiddleware
        gate = GeneralBudgetMiddleware()
        result = gate.before_model({"executor_model_run_limit": 8,
            "executor_model_calls_used": 2, "worker_finalize_requested": True,
            "worker_finalization_model_calls_used": 1}, None)
        self.assertEqual(result, {"jump_to": "end"})

    async def run_planning(
        self, *, status="COMPLETED", natural=False, artifact=False, no_budget=False, approve=True
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            async with AsyncEventStore(root / "events.sqlite3") as store:
                model = GeneralModel(
                    status=status, natural_exit=natural, write_file=artifact
                )
                scheduler = SchedulerModel(artifact=artifact, approve=approve)

                def factory(*args, **kwargs):
                    return create_general_worker(*args, **kwargs)

                runtime = GeneralStepRuntime(
                    model,
                    event_store=store,
                    tools=[business_probe],
                    progress_every_tool_calls=4,
                    run_storage_root=root / "runs",
                    worker_factory=factory,
                )

                class RuntimeInput:
                    async def ainvoke(self, input_state, config=None):
                        if artifact:
                            input_state = {
                                **input_state,
                                "files": {
                                    "/artifacts/note.md": create_file_data(
                                        "actual note"
                                    )
                                },
                            }
                        return await runtime.ainvoke(input_state, config=config)

                settings = planning_settings()
                # Deliberately permit retries globally: General must still run once.
                from dataclasses import replace

                settings = replace(settings, max_step_attempts=3)
                if no_budget:
                    settings = replace(settings, max_step_executor_rounds=0)
                graph = build_planning_graph(
                    simple_model=scheduler,
                    hard_model=scheduler,
                    worker_registry=WorkerAgentRegistry({"GENERAL": RuntimeInput()}),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=settings,
                    model_output_max_tokens=2048,
                    run_storage_root=root / "runs",
                )

                async def progress(event):
                    pass

                final = None
                nodes = []
                async for mode, value in graph.astream(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request="Original observation request",
                        ),
                        "event_id": "event",
                        "planning_run_id": "event",
                        "conversation_thread_id": "conv",
                    },
                    config={"configurable": {"progress_callback": progress}},
                    stream_mode=["updates", "values"],
                ):
                    if mode == "updates":
                        nodes.extend(value)
                    else:
                        final = value
                if not natural and not no_budget:
                    self.assertEqual(final["completed_step_reports"][0].handoff_knowledge[0].topic, "probe")
                else:
                    self.assertEqual(final["completed_step_reports"][0].handoff_knowledge, [])
                self.assertIn("general_report", nodes)
                self.assertNotIn("step_reporter", nodes)
                self.assertNotIn("code_controller", nodes)
                needs_review = status != "COMPLETED" or natural or artifact or no_budget
                self.assertEqual("StepReport" in scheduler.schemas, needs_review)
                self.assertEqual(len(model.requests), 0 if no_budget else 2)
                report = final["completed_step_reports"][0]
                self.assertEqual(
                    report.assessment_source,
                    "INDEPENDENT_REVIEW" if needs_review else "GENERAL_SELF_REPORT",
                )
                self.assertEqual(final["current_step_report_rounds"], 1 if needs_review else 0)
                if artifact and approve:
                    self.assertEqual(len(report.artifacts), 1)
                    receipt = final["handoff_publication_receipts"][0]
                    self.assertEqual(
                        Path(receipt["storage_path"]).read_text(), "actual note"
                    )
                elif artifact:
                    self.assertEqual(report.artifacts, [])
                    self.assertFalse(final.get("handoff_publication_receipts"))
                    self.assertNotEqual(report.status, "COMPLETED")
                return report

    async def test_completed_general_result_skips_independent_reporter(self):
        report = await self.run_planning()
        self.assertEqual(report.status, "COMPLETED")
        self.assertEqual(report.confirmed_results, [])

    async def test_blocked_general_does_not_enter_retry_loop(self):
        report = await self.run_planning(status="BLOCKED")
        self.assertEqual(report.status, "BLOCKED")

    async def test_unstructured_exit_is_partial_without_reformat_call(self):
        report = await self.run_planning(natural=True)
        self.assertEqual(report.status, "PARTIAL")

    async def test_general_files_require_independent_report_before_shared_handoff(self):
        await self.run_planning(artifact=True)

    async def test_general_file_is_not_shared_without_reporter_approval(self):
        await self.run_planning(artifact=True, approve=False)

    async def test_unstarted_general_with_no_budget_skips_review(self):
        report = await self.run_planning(no_budget=True)
        self.assertEqual(report.status, "FAILED")

    def test_completion_schema_hides_runtime_and_keeps_file_schema_small(self):
        schema = convert_to_openai_tool(report_general_result)["function"]["parameters"]
        self.assertNotIn("runtime", schema["properties"])
        self.assertNotIn("repository_url", json.dumps(schema))
        self.assertTrue(
            all(field.description for field in GeneralResult.model_fields.values())
        )

    async def test_simple_agent_still_cannot_overwrite_shared_input(self):
        from run_workspace import (
            create_run_worker_backend,
            handoff_read_only_permissions,
            initialize_run_workspace,
        )

        class WriteModel(GeneralModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                if any(isinstance(m, ToolMessage) for m in messages):
                    message = AIMessage(content="Cannot overwrite input.")
                else:
                    message = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "write",
                                "name": "edit_file",
                                "args": {
                                    "source_refs": ["U1"],
                                    "file_path": "/handoff/input.txt",
                                    "old_string": "original",
                                    "new_string": "changed",
                                },
                            }
                        ],
                    )
                return ChatResult(generations=[ChatGeneration(message=message)])

        with TemporaryDirectory() as directory:
            layout = initialize_run_workspace("readonly", storage_root=Path(directory))
            original = layout.handoff_root / "input.txt"
            original.write_text("original")
            graph = create_general_worker(
                WriteModel(),
                tools=[],
                backend=create_run_worker_backend(layout),
                permissions=handoff_read_only_permissions(),
            )
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="Try writing input")]}
            )
            self.assertEqual(original.read_text(), "original")
            messages = [
                m.content for m in result["messages"] if isinstance(m, ToolMessage)
            ]
            self.assertTrue(any("denied" in str(m).lower() for m in messages), messages)
