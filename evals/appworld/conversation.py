"""Opt-in AppWorld boundary around the normal ConversationRuntime.

No schema/provider/callback replacement. One adapter owns exactly one task world.
The caller owns world.close() and runtime.stop(), including error paths.
"""
from __future__ import annotations

import threading
from uuid import uuid4

from langchain_core.tools import ToolException, tool

from conversation_runtime import ConversationRuntime
from eventing import build_planning_thread_id_from_parts
from observability import trace_span, set_span_output
from trace_overview import task_overview
from skill_runtime import load_catalog
from evals.appworld.tool_descriptions import (
    DISCOVER_DESCRIPTION, EXECUTE_DESCRIPTION, VERIFY_DESCRIPTION,
    AppWorldCallInput, AppWorldDiscoverInput,
)
from workers.execution_state import ExecutionStateLedger, execution_state_scope


def appworld_skill_catalog(catalog):
    """Exclude network-routing strategies only in the opt-in benchmark context."""
    return [asset for asset in catalog
            if asset["source"] != "web"
            and not set(asset.get("topics", ())).intersection({"web", "web_search", "browsing"})
            and asset["name"] not in {"plan-web-research", "plan-task-dependencies"}]

class TaskTools:
    """Serialize all roles against one world; stop accepting calls before judging."""

    def __init__(self, world):
        self.world = world
        self.lock = threading.RLock()
        self.accepting = True
        self.calls = []
        self.execution_state = ExecutionStateLedger()

    def _record_execution_state(self, **facts):
        """State hints are best-effort and may never change business execution."""
        try:
            self.execution_state.record(**facts)
            return self.execution_state.snapshot()
        except Exception:
            return None

    @staticmethod
    def _validate_discovery(code: str):
        import re
        calls = re.findall(
            r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)", code
        )
        if not calls or any(app != "api_docs" for app, _ in calls):
            raise ToolException(
                "appworld_discover only permits apis.api_docs.* documentation calls; "
                "use appworld_execute after discovering the exact business API."
            )

    def execute(self, code, role, reason=None, source_refs=None, action_phase="OTHER", binding_ref=None, set_bindings=None):
        with self.lock:
            if not self.accepting:
                raise ToolException("This task has ended; AppWorld execution is locked.")
            record = {"index": len(self.calls) + 1, "role": role, "reason": reason,
                      "source_refs": list(source_refs or ()), "action_phase": action_phase,
                      "binding_ref": binding_ref,
                      "set_bindings": list(set_bindings or ()), "code": code}
            self.calls.append(record)
            try:
                output = self.world.execute(code)
                record["output"] = output
                if isinstance(output, str) and output.lstrip().startswith("Execution failed"):
                    raise ToolException(output)
                state = self._record_execution_state(
                    index=record["index"], role=role, code=code, succeeded=True
                )
                if state is not None:
                    record["execution_state"] = state
                return output
            except Exception as error:
                record["error"] = str(error)
                state = self._record_execution_state(
                    index=record["index"], role=role, code=code,
                    succeeded=False, error_type=type(error).__name__,
                )
                if state is not None:
                    record["execution_state"] = state
                raise

    def build(self):
        @tool(description=DISCOVER_DESCRIPTION, args_schema=AppWorldDiscoverInput)
        def appworld_discover(source_refs: list[str], reason: str, code: str) -> str:
            """Read exact AppWorld API documentation without business operations."""
            self._validate_discovery(code)
            return self.execute(code, "discover", reason, source_refs)

        @tool(description=EXECUTE_DESCRIPTION, args_schema=AppWorldCallInput)
        def appworld_execute(source_refs: list[str], reason: str, code: str, action_phase: str = "OTHER", binding_ref: str | None = None, binding_checks: list[dict] | None = None, set_bindings: list[dict] | None = None) -> str:
            """Run Python with documented apis in this task's persistent AppWorld.

            Print results. Docker files/variables are separate; send code explicitly.
            This is for General and Code Worker. Never access hidden evaluator data.
            """
            return self.execute(code, "executor", reason, source_refs, action_phase, binding_ref, set_bindings)

        @tool(description=VERIFY_DESCRIPTION, args_schema=AppWorldCallInput)
        def appworld_verify(source_refs: list[str], reason: str, code: str, action_phase: str = "OTHER", binding_ref: str | None = None, binding_checks: list[dict] | None = None, set_bindings: list[dict] | None = None) -> str:
            """Independently query this same AppWorld for Code Reviewer verification.

            Use documented APIs and explicit inputs. Do not repeat business writes.
            Python access is not technically read-only. Repair belongs to Code Worker.
            """
            return self.execute(code, "code_reviewer", reason, source_refs, action_phase, binding_ref, set_bindings)

        # Canonical LangChain callbacks own the tool spans; do not add duplicate spans.
        return appworld_discover, appworld_execute, appworld_verify


def can_judge(snapshot):
    values = snapshot.values
    return (not snapshot.next
            and (
                values.get("scheduler_final_decision") is True
                or values.get("harness_terminal_decision") is True
            )
            and values.get("final_status") in {"COMPLETED", "FAILED", "PARTIAL"})


class AppWorldConversation(ConversationRuntime):
    """One prefixed benchmark user request; planning methods use normal Skills."""

    def __init__(self, settings, world, task):
        self.task = dict(task)
        self.task_tools = TaskTools(world)
        self.discover_tool, self.execute_tool, self.verify_tool = self.task_tools.build()
        self._task_started = False
        self._completion_api_contract = ""
        super().__init__(settings, [self.discover_tool, self.execute_tool])

    async def start(self):
        await super().start()
        # Code agents are built lazily for a Step, after start and before ask.
        self.code_runtime.tools = (self.discover_tool, self.execute_tool)
        self.code_runtime.reviewer_tools = (self.discover_tool, self.verify_tool)

    async def _prepare_planning_context(self, **kwargs):
        context = await super()._prepare_planning_context(**kwargs)
        return context.model_copy(update={
            "skill_catalog": appworld_skill_catalog(
                context.skill_catalog if context.skill_catalog is not None else load_catalog()),
            "current_time": self.task["datetime"],
            "execution_environment": "appworld",
            "completion_api_contract": self._completion_api_contract,
            "toolset_catalog": [{"name": "APPWORLD", "description":
                "在当前任务的隔离 AppWorld 模拟环境中，先用appworld_discover查精确接口名和签名，"
                "再用appworld_execute查询或修改数据；Code Reviewer用appworld_discover和appworld_verify独立核验。"}],
        })

    def _read_completion_api_contract(self) -> str:
        """Read the public completion signature, never the evaluator or task answer."""
        code = ('print(apis.api_docs.show_api_doc('
                'app_name="supervisor", api_name="complete_task"))')
        self.task_tools._validate_discovery(code)
        document = self.task_tools.execute(
            code, "harness_documentation",
            reason="Read the public completion API contract once for role handoff.",
            source_refs=["api_docs.supervisor.complete_task"],
        )
        if not isinstance(document, str) or not all(
            term in document for term in ("complete_task", "answer", "status")
        ) or len(document) > 20000:
            raise RuntimeError("AppWorld did not return a bounded complete_task API document")
        return (
            "[AppWorld public API documentation: supervisor.complete_task]\n"
            + document.strip()
            + "\nThis is the public API contract, not a hidden grading rule or proof of task success."
        )

    async def _consolidate_memory_background(self, **kwargs):
        # Benchmark task text is not durable user memory. The original runtime
        # buffers it even with extraction disabled, creating an unrelated trace.
        # Normal conversations retain their existing memory workflow.
        return None

    async def run_task(self, progress_callback=None):
        if self._task_started:
            raise RuntimeError("Create a new adapter/world for each task; reuse is forbidden.")
        self._task_started = True
        if progress_callback is None:
            async def progress_callback(event):
                pass
        task_id = self.task["task_id"]
        event_id = "appworld-" + uuid4().hex
        conversation = await self.new_conversation(
            "appworld", event_id, title="AppWorld " + task_id)
        thread_id = build_planning_thread_id_from_parts(
            conversation_id=conversation.thread_id, event_id=event_id)
        result = {"task_id": task_id, "conversation_id": conversation.conversation_id,
                  "event_id": event_id, "planning_thread_id": thread_id,
                  "official_evaluation": None}
        try:
            with trace_span("AppWorld / " + task_id, attributes={"appworld.task_id": task_id,
                    "session.id": conversation.conversation_id}) as span, task_overview(span, self.task, result):
                self._completion_api_contract = self._read_completion_api_contract()
                with execution_state_scope(self.task_tools.execution_state):
                    result["answer"] = await self.ask(
                        user_text="当前我们正在 AppWorld 里面进行测试。\n\n" + self.task["instruction"], channel="appworld",
                        external_chat_id=event_id, event_id=event_id,
                        target_conversation_id=conversation.conversation_id,
                        progress_callback=progress_callback)
                snapshot = await self.planning_graph.aget_state(
                    {"configurable": {"thread_id": thread_id}})
                result["scheduler_status"] = snapshot.values.get("final_status")
                result["stop_reason"] = snapshot.values.get("overall_stop_reason")
                result["terminal_decision_source"] = (
                    "scheduler_model"
                    if snapshot.values.get("scheduler_final_decision") is True
                    else (
                        "harness"
                        if snapshot.values.get("harness_terminal_decision") is True
                        else None
                    )
                )
                with self.task_tools.lock:
                    self.task_tools.accepting = False
                    if can_judge(snapshot):
                        self.task_tools.world.request("finish")
                        with trace_span("AppWorld Judge") as judge:
                            result["official_evaluation"] = self.task_tools.world.request("evaluate")
                            if judge is not None:
                                judge.set_attribute("business.status", "PASSED" if result["official_evaluation"].get("success") else "FAILED")
                            set_span_output(judge, result["official_evaluation"])
                    else:
                        result["evaluation_skipped"] = "No terminal Scheduler Final Review; count as an unfinished trial."
                if span is not None:
                    span.set_attribute("business.status", result["scheduler_status"] or "UNKNOWN")
                    span.set_attribute("appworld.evaluation_status", "SKIPPED" if result["official_evaluation"] is None else
                                       ("PASSED" if result["official_evaluation"].get("success") else "FAILED"))
                set_span_output(span, result)
            return result
        finally:
            self.task_tools.accepting = False
