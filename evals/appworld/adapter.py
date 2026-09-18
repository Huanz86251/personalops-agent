"""Reuse PersonalOps' supervisor/executor/reporter/replanner graph for AppWorld."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import re
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import ToolException, tool

from config import PLANNING_INTEGER_OPTIONS, PlanningSettings
from eventing import AsyncEventStore
from middlewares import DynamicExecutionBudgetMiddleware
from planning_graph import build_planning_graph
from planning_models import PlanningContextPack
from workers.general_completion import GENERAL_REPORT_NAME
from workers import (
    WorkerAgentRegistry,
    WorkerGroupCoordinator,
    PUBLISH_WORKER_PROGRESS_NAME,
    WorkerLeadershipBridge,
    WorkerToolAllowlistMiddleware,
    create_general_worker,
)

ENVIRONMENT_INSTRUCTIONS = """You are operating only inside a simulated AppWorld.
appworld_discover and appworld_execute are available in one persistent task world.
Use print(...) to observe results. Variables persist across calls and planning steps.
Use appworld_discover for documentation only. Discover available apps with print(apis.api_docs.show_app_descriptions()).
List an app's APIs with print(apis.api_docs.show_api_descriptions(app_name="APP_NAME")).
Read a signature with print(apis.api_docs.show_api_doc(app_name="APP_NAME", api_name="API_NAME")).
All API arguments are keyword arguments; use these documentation APIs, not Python introspection.
Use appworld_execute for business queries and writes after checking the cited source and signature.
The current time and simulated account owner are provided in the task context.
Use the supervisor APIs to obtain account information as documented.
Use only documented apis; never inspect evaluator/ground-truth/internal files or modules.
Calls modifying simulated records for this user request are authorized.
Do not request host files, shell tools, browser tools, external network, or real accounts.
Combine dependent operations into one code call; do not issue parallel code calls.
When the entire user request is fulfilled, call apis.supervisor.complete_task()
with the documented arguments, including an answer when the task requires one.
Do not mark the whole task complete after merely finishing an intermediate planning step.
"""


class EvaluationBudgetExceeded(RuntimeError):
    pass


def _trace_message(message):
    """Serialize the complete LangChain message for local trace inspection."""

    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump(mode="json")
        except TypeError:
            return message.model_dump()
    return {
        "message_type": message.__class__.__name__,
        "value": str(message),
    }


def _trace_messages(batches):
    messages = []
    for batch in batches or []:
        if isinstance(batch, list):
            messages.extend(_trace_message(message) for message in batch)
        else:
            messages.append(_trace_message(batch))
    return messages


class UsageMeter(BaseCallbackHandler):
    """One record per LangChain model invocation, independent of OTel layering."""
    raise_error = True
    # AsyncCallbackManager constructs every inline callback coroutine before it
    # awaits them one by one.  Raising the hard budget error from an inline
    # callback therefore strands any later coroutine and emits ``was never
    # awaited`` during cleanup.  Keep the pre-send rejection, but let LangChain
    # await it in the gathered callback group so sibling callbacks are drained.
    run_inline = False

    def __init__(self, max_calls=80):
        self.max_calls = max_calls
        self.started = {}
        self.records = {}
        self.call_metadata = {}
        self.trace_spans = {}
        self.lock = threading.Lock()

    @staticmethod
    def _model_role(tags, metadata=None):
        role = (metadata or {}).get("runtime.model_role")
        if isinstance(role, str) and role.strip():
            return role.strip()
        for tag in tags or []:
            if isinstance(tag, str) and tag.startswith("appworld:"):
                return tag.split(":", 1)[1]
        return "unclassified"

    def _start_trace_span(
        self,
        key,
        call_index,
        role,
        model_name,
        messages,
        serialized,
        invocation_params,
    ):
        try:
            from observability import get_tracer, set_span_attributes, set_span_input
            tracer = get_tracer()
            if tracer is None:
                return
            span = tracer.start_span(
                f"LLM / {role} / call {call_index:02d}",
                openinference_span_kind="llm",
            )
            traced_messages = _trace_messages(messages)
            set_span_input(span, {
                "reading_guide": (
                    "Messages actually sent to the model, in order."
                ),
                "message_count": len(traced_messages),
                "messages": traced_messages,
                "model_serialized": serialized,
                "invocation_parameters": invocation_params,
            })
            set_span_attributes(
                span,
                **{
                    "eval.call_index": call_index,
                    "eval.model_role": role,
                    "llm.model_name": model_name or "unknown",
                },
            )
            self.trace_spans[key] = span
        except Exception:
            # Observability must never change model execution.
            return

    def _finish_trace_span(self, span, record, output_message=None, error=None):
        if span is None:
            return
        try:
            from observability import set_span_attributes, set_span_output
            from opentelemetry.trace import Status, StatusCode
            set_span_attributes(
                span,
                **{
                    "eval.model_status": record.get("status"),
                    "llm.token_count.prompt": record.get("input_tokens"),
                    "llm.token_count.completion": record.get("output_tokens"),
                    "llm.finish_reason": record.get("finish_reason") or "unknown",
                    "eval.usage_source": record.get("usage_source") or "unknown",
                },
            )
            set_span_output(span, {
                "reading_guide": (
                    "Model response. tool_calls is the model's tool decision."
                ),
                "message": (
                    _trace_message(output_message)
                    if output_message is not None
                    else None
                ),
                "status": record.get("status"),
                "finish_reason": record.get("finish_reason"),
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
            })
            if error is not None:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            else:
                span.set_status(Status(StatusCode.OK))
        finally:
            span.end()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.lock:
            key = str(run_id)
            if key in self.started:
                return
            if len(self.started) >= self.max_calls:
                if (kwargs.get("metadata") or {}).get("trace.owner") == "personalops":
                    from trace_callbacks import CALLBACK
                    CALLBACK.reject_before_send(run_id, "Trial model-call budget exhausted")
                raise EvaluationBudgetExceeded("Trial model-call budget exhausted")
            self.started[key] = time.monotonic()
            tags = kwargs.get("tags", [])
            model_name = (kwargs.get("invocation_params") or {}).get("model_name")
            role = self._model_role(tags, kwargs.get("metadata"))
            call_index = len(self.started)
            self.call_metadata[key] = {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "tags": tags,
                "model_name": model_name,
                "model_role": role,
                "call_index": call_index,
            }
            # Runtime models own the canonical leaf; keep evaluation accounting
            # without publishing a second LLM span for the same request.
            if (kwargs.get("metadata") or {}).get("trace.owner") == "personalops":
                return
            self._start_trace_span(
                key,
                call_index,
                role,
                model_name,
                messages,
                serialized,
                kwargs.get("invocation_params") or {},
            )

    def on_llm_end(self, response, *, run_id, **kwargs):
        key = str(run_id)
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        response_metadata = getattr(message, "response_metadata", {}) or {}
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            raw = (response.llm_output or {}).get("token_usage", {})
            usage = {"input_tokens": raw.get("prompt_tokens"),
                     "output_tokens": raw.get("completion_tokens")}
        with self.lock:
            if key in self.records:
                return
            record = {
                **self.call_metadata.get(key, {}),
                "model_name": response_metadata.get("model_name") or self.call_metadata.get(key, {}).get("model_name"),
                "provider_response_id": response_metadata.get("id"),
                "system_fingerprint": response_metadata.get("system_fingerprint"),
                "cache_read_input_tokens": (usage.get("input_token_details") or {}).get("cache_read"),
                "reasoning_output_tokens": (usage.get("output_token_details") or {}).get("reasoning"),
                "finish_reason": response_metadata.get("finish_reason"),
                "usage_source": "model_callback",
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "latency_seconds": time.monotonic() - self.started.get(key, time.monotonic()),
                "status": "returned",
            }
            self.records[key] = record
            trace_span = self.trace_spans.pop(key, None)
        self._finish_trace_span(trace_span, record, output_message=message)

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            if str(run_id) not in self.started:
                return
            key = str(run_id)
            if key in self.records:
                return
            # SDK parsing errors can contain a completed, billable provider response.
            # Use its structured usage, never extract numbers from exception prose.
            completion = getattr(error, "completion", None)
            if hasattr(completion, "model_dump"):
                completion = completion.model_dump()
            completion = completion if isinstance(completion, dict) else {}
            raw = completion.get("usage") or {}
            choices = completion.get("choices") or []
            record = {
                **self.call_metadata.get(key, {}),
                "status": "error", "error_type": type(error).__name__,
                "model_name": completion.get("model") or self.call_metadata.get(key, {}).get("model_name"),
                "provider_response_id": completion.get("id"),
                "system_fingerprint": completion.get("system_fingerprint"),
                "input_tokens": raw.get("prompt_tokens"),
                "output_tokens": raw.get("completion_tokens"),
                "cache_read_input_tokens": (raw.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "reasoning_output_tokens": (raw.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "usage_source": "provider_response_on_error" if raw else "unknown",
                "latency_seconds": time.monotonic() - self.started[key],
            }
            self.records[key] = record
            trace_span = self.trace_spans.pop(key, None)
        output_message = choices[0].get("message") if choices else None
        self._finish_trace_span(
            trace_span,
            record,
            output_message=output_message,
            error=error,
        )

    def report(self):
        with self.lock:
            records = list(self.records.values())
            complete = (len(records) == len(self.started)
                        and all(r.get("input_tokens") is not None
                                and r.get("output_tokens") is not None for r in records))
            return {
                "model_calls_started": len(self.started),
                "model_calls_returned": sum(r["status"] == "returned" for r in records),
                "usage_complete": complete,
                "known_input_tokens": sum(r.get("input_tokens") or 0 for r in records),
                "known_output_tokens": sum(r.get("output_tokens") or 0 for r in records),
                "input_tokens": sum(r["input_tokens"] for r in records) if complete else None,
                "output_tokens": sum(r["output_tokens"] for r in records) if complete else None,
                "records": records,
            }


def default_planning():
    # Fixed code defaults, not the user's live .env or personal session settings.
    return PlanningSettings(**{name: spec[1] for name, spec in PLANNING_INTEGER_OPTIONS.items()})


def make_discover_tool(world):
    from evals.appworld.tool_descriptions import DISCOVER_DESCRIPTION, AppWorldDiscoverInput

    @tool(description=DISCOVER_DESCRIPTION, args_schema=AppWorldDiscoverInput)
    def appworld_discover(source_refs: list[str], reason: str, code: str) -> str:
        calls = re.findall(
            r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)", code
        )
        if not calls or any(app != "api_docs" for app, _ in calls):
            raise ToolException(
                "appworld_discover only permits apis.api_docs.* documentation calls; "
                "use appworld_execute after discovering the exact business API."
            )
        output = world.execute(code)
        if output.startswith("Execution failed."):
            raise ToolException(output)
        return output

    appworld_discover.handle_tool_error = True
    return appworld_discover


def make_execute_tool(world):
    tool_call_index = 0
    from evals.appworld.tool_descriptions import EXECUTE_DESCRIPTION, AppWorldCallInput

    @tool(description=EXECUTE_DESCRIPTION, args_schema=AppWorldCallInput)
    def appworld_execute(source_refs: list[str], reason: str, code: str, action_phase: str = "OTHER", binding_ref: str | None = None, binding_checks: list[dict] | None = None, set_bindings: list[dict] | None = None) -> str:
        """Execute Python in the persistent simulated AppWorld; print results to observe them.
        Use documented apis only. No host filesystem, shell or external network access.
        Submit one code call at a time; variables persist. Never access grading internals.
        """
        from observability import trace_span, set_span_attributes, set_span_output

        nonlocal tool_call_index
        tool_call_index += 1
        api_calls = sorted(set(
            ".".join(match)
            for match in re.findall(
                r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
                code,
            )
        ))
        with trace_span(
            f"TOOL / AppWorld execute / call {tool_call_index:02d}",
            kind="tool",
            input_value={
                "reading_guide": (
                    "Tool selected by the model and the exact Python "
                    "it asked AppWorld to run."
                ),
                "reason": reason,
                "source_refs": source_refs,
                "action_phase": action_phase,
                "binding_ref": binding_ref,
                "binding_checks": list(binding_checks or ()),
                "set_bindings": list(set_bindings or ()),
                "code": code,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "code_chars": len(code),
                "code_lines": len(code.splitlines()),
                "api_calls": api_calls[:20],
            },
        ) as span:
            output = world.execute(code)
            failed = output.startswith("Execution failed.")
            set_span_attributes(
                span,
                **{
                    "eval.tool_failed": failed,
                    "eval.api_call_count": len(api_calls),
                    "eval.output_chars": len(output),
                },
            )
            set_span_output(span, {
                "reading_guide": (
                    "Exact AppWorld observation returned to the agent "
                    "as a tool message for the next model round."
                ),
                "status": "error" if failed else "returned",
                "observation": output,
                "output_chars": len(output),
            })
            if failed:
                raise ToolException(output)
            return output
    appworld_execute.handle_tool_error = True
    return appworld_execute


async def run_personalops(world, task, *, simple_model, hard_model, trial_id,
                          skill="", planning=None, output_max_tokens=2048, wall_timeout=600,
                          skill_mode="off", skill_fixed_ids=None,
                          checkpointer=None, checkpoint_thread_id=None):
    planning = planning or default_planning()
    instructions = ENVIRONMENT_INSTRUCTIONS
    if skill.strip():
        instructions += "\nAdditional reusable skill:\n" + skill.strip()
    discover = make_discover_tool(world)
    execute = make_execute_tool(world)
    progress_store = AsyncEventStore()
    await progress_store.start()
    worker_graph = create_general_worker(
        simple_model,
        tools=[discover, execute],
        middleware=[
            DynamicExecutionBudgetMiddleware(),
            WorkerToolAllowlistMiddleware(
                {
                    "appworld_discover",
                    "appworld_execute",
                    GENERAL_REPORT_NAME,
                }
            ),
        ],
        checkpointer=checkpointer,
        store=None,
    )
    step_worker = WorkerLeadershipBridge(worker_graph, progress_store)
    graph = build_planning_graph(
        simple_model=simple_model, hard_model=hard_model,
        worker_registry=WorkerAgentRegistry.general_worker_first(step_worker),
        worker_group_coordinator=WorkerGroupCoordinator(progress_store),
        planning=planning, model_output_max_tokens=output_max_tokens,
        checkpointer=checkpointer,
    )
    context = PlanningContextPack(
        skill_mode=skill_mode, skill_fixed_ids=skill_fixed_ids or {},
        current_time=task["datetime"], execution_environment="appworld",
        user_request=task["instruction"],
        execution_instructions=instructions + "\nSimulated account owner:\n"
                               + json.dumps(task["supervisor"], ensure_ascii=False),
        toolset_catalog=[{"name": "APPWORLD", "description":
            "先用appworld_discover确认接口名和签名，再用appworld_execute执行。"}],
    )
    progress_stages = []

    async def record_progress(event):
        progress_stages.append(event.stage)

    configurable = {"progress_callback": record_progress}
    invoke_options = {}
    if checkpointer is not None:
        configurable["thread_id"] = checkpoint_thread_id or f"planning:{trial_id}"
        invoke_options["durability"] = "sync"
    try:
        result = await asyncio.wait_for(graph.ainvoke({
            "context": context,
            "event_id": trial_id,
            "conversation_thread_id": trial_id,
            "planning_run_id": trial_id,
        }, config={"configurable": configurable}, **invoke_options), timeout=wall_timeout)
    finally:
        await progress_store.close()
    return {
        "self_reported_final_status": result.get("final_status"),
        "progress_stages": progress_stages,
        "stop_reason": result.get("overall_stop_reason"),
        "terminal_decision_source": (
            "scheduler_model"
            if result.get("scheduler_final_decision") is True
            else (
                "harness"
                if result.get("harness_terminal_decision") is True
                else None
            )
        ),
        "reported_model_rounds": result.get("model_rounds_used"),
        "reported_tool_calls": result.get("tool_calls_used"),
        "final_answer": result.get("final_answer"),
        "step_reports": [r.model_dump(mode="json") for r in result.get("completed_step_reports", [])],
        "planning": asdict(planning),
        "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(),
        "skill_routing_mode": skill_mode,
        "skill_routing_snapshots": result["context"].role_skill_snapshots,
        "skill_catalog": result["context"].skill_catalog,
        "tool_schema_sha256": hashlib.sha256(json.dumps({
            "name": execute.name, "description": execute.description,
            "schema": execute.args_schema.model_json_schema(),
        }, sort_keys=True).encode()).hexdigest(),
    }
