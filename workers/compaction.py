"""Compact old execution chatter, preserving assignments and control handoffs."""
import json
import hashlib

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from prompt_loader import load_prompt
from schemas.compaction import WorkerExecutionSummary
from structured_output_policy import textual_schema_fallback
from compaction_runtime import CompactionFailure, run_compaction, arun_compaction
from config import load_worker_compaction_settings
from langchain.agents.middleware.types import ExtendedModelResponse
from langgraph.types import Command
from langchain_core.utils.function_calling import convert_to_openai_tool


CONTROL = {"submit_code_for_review", "respond_to_code_review",
           "request_code_worker_repair", "publish_reviewed_candidate", "submit_code_review",
           "complete_web_step", "report_general_result"}


class WorkerCompactionMiddleware(AgentMiddleware):
    # Replaces DeepAgents' generic summarizer, rather than stacking two policies.
    name = "SummarizationMiddleware"

    def __init__(
        self,
        model,
        threshold=None,
        keep_messages=None,
        keep_tokens=None,
        enabled=None,
        max_chars=None,
        request_aware=False,
    ):
        if isinstance(model, str):
            from langchain.chat_models import init_chat_model
            model = init_chat_model(model)
        self.model = model
        # Dedicated summary clone: do not inherit the role's generation cap.
        if hasattr(model, "model_copy") and hasattr(model, "max_tokens"):
            kwargs = dict(getattr(model, "model_kwargs", {}) or {})
            kwargs.pop("max_tokens", None)
            kwargs.pop("max_completion_tokens", None)
            self.model = model.model_copy(update={"max_tokens": None, "model_kwargs": kwargs})
        settings = load_worker_compaction_settings()
        self.settings = settings
        self.request_aware = request_aware
        if request_aware:
            # Unique name prevents DeepAgents replacing its early default slot.
            self.name = "WorkerExecutionCompactionMiddleware"
        self.enabled = (
            bool(enabled)
            if enabled is not None
            else settings.enabled
        )
        self.threshold = threshold if threshold is not None else settings.trigger_tokens
        self.keep_messages = (
            int(keep_messages)
            if keep_messages is not None
            else settings.keep_messages
        )
        self.keep_tokens = (
            int(keep_tokens)
            if keep_tokens is not None
            else settings.keep_tokens
        )
        self.max_chars = max_chars if max_chars is not None else settings.max_chars
        if self.max_chars < 1 or self.threshold < 1:
            raise ValueError("Worker compaction limits must be positive.")
        if self.keep_messages < 1 or self.keep_tokens < 1:
            raise ValueError("Worker compaction retention must be positive.")

    def _recent_boundary(self, messages):
        """Keep a token-bounded tail without splitting a tool exchange."""

        if not messages:
            return 0
        boundary = max(0, len(messages) - self.keep_messages)
        retained_tokens = count_tokens_approximately(messages[boundary:])
        while boundary > 0 and retained_tokens < self.keep_tokens:
            boundary -= 1
            retained_tokens += count_tokens_approximately([messages[boundary]])

        # If the boundary lands inside an AI tool-call/result group, keep the
        # whole group. Tool results must never be orphaned from their call.
        while boundary > 0 and getattr(messages[boundary], "type", "") == "tool":
            boundary -= 1
        return boundary

    def _select(self, state, *, ignore_threshold=False):
        messages = list(state.get("messages", []))
        if not ignore_threshold and count_tokens_approximately(messages) < self.threshold:
            return messages, []
        # Keep a recent 4k-token tail (at least two messages), every
        # original/runtime instruction, and complete control exchanges.
        boundary = self._recent_boundary(messages)
        selected = []
        index = 0
        while index < boundary:
            message = messages[index]
            if getattr(message, "additional_kwargs", {}).get("worker_compaction_summary"):
                selected.append(index)
                index += 1
                continue
            if getattr(message, "additional_kwargs", {}).get("knowledge_source"):
                selected.append(index)
                index += 1
                continue
            if getattr(message, "type", "") != "ai":
                index += 1
                continue
            calls = getattr(message, "tool_calls", [])
            if not calls:
                selected.append(index)
                index += 1
                continue
            end = index + 1
            while end < boundary and getattr(messages[end], "type", "") == "tool":
                end += 1
            expected = {call["id"] for call in calls}
            actual = {getattr(item, "tool_call_id", None) for item in messages[index + 1:end]}
            if expected == actual and not any(call["name"] in CONTROL for call in calls):
                selected.extend(range(index, end))
            index = end
        return messages, selected

    def _request(self, messages, selected, refs=None, budget=None):
        from workers.evidence_refs import display
        previous, execution = [], []
        for index in selected:
            message = messages[index]
            if getattr(message, "additional_kwargs", {}).get("worker_compaction_summary"):
                previous.append(json.loads(message.content.split("\n", 1)[1]))
                continue
            item = message.model_dump(mode="json")
            for field in ("additional_kwargs", "response_metadata", "usage_metadata"):
                item.pop(field, None)
            execution.append(item)
        return [
            {"role": "system", "content": load_prompt("conversation/worker_summary")
             + f"\n输出JSON不超过{self.max_chars}个字符。"
             + (f"\n本次摘要目标不超过约{budget['summary_tokens']} token（包含JSON包装），不是字符数。" if budget else "")
             + textual_schema_fallback(self.model, WorkerExecutionSummary)},
            {"role": "user", "content": json.dumps({
                "previous_summaries": previous,
                "execution_messages": display(execution, refs or {}),
            }, ensure_ascii=False, separators=(",", ":"))},
        ]

    def _summary_message(self, summary):
        return HumanMessage(content="执行过程JSON摘要：\n" + summary.model_dump_json(),
            additional_kwargs={"personalops_runtime_event": True, "worker_compaction_summary": True})

    def _check(self, messages, selected, refs, budget=None):
        selected_refs = set()
        for index in selected:
            message = messages[index]
            for call in getattr(message, "tool_calls", []) or []:
                if call.get("id") in refs:
                    selected_refs.add(refs[call["id"]])
            call_id = getattr(message, "tool_call_id", None)
            if call_id in refs:
                selected_refs.add(refs[call_id])
            if getattr(message, "additional_kwargs", {}).get("worker_compaction_summary"):
                old = json.loads(message.content.split("\n", 1)[1])
                selected_refs.update(old.get("evidence_refs", []))
        known = set(refs.values())
        selected_refs &= known

        def check(summary):
            # Reference identity comes from the Harness. Do not trust invented IDs.
            summary.evidence_refs = sorted(set(summary.evidence_refs) & selected_refs)
            if len(summary.model_dump_json()) > self.max_chars:
                raise ValueError("summary_size_exceeded")
            if count_tokens_approximately([self._summary_message(summary)]) >= count_tokens_approximately(
                [messages[i] for i in selected]
            ):
                raise ValueError("summary_did_not_shrink")
            if budget:
                size = count_tokens_approximately([self._summary_message(summary)])
                if size > budget['summary_tokens'] or size + budget['protected_tokens'] > self.settings.target_tokens - self.settings.safety_tokens:
                    raise ValueError(f"summary_budget_exceeded:actual={size},target={budget['summary_tokens']}")
        return check

    def _finish(self, messages, selected, summary):
        chosen = set(selected)
        retained = []
        for index, message in enumerate(messages):
            if index == selected[0]:
                retained.append(self._summary_message(summary))
            if index not in chosen:
                retained.append(message)
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *retained],
                "_archived": [messages[index] for index in selected]}

    def _allowed(self, state):
        used = sum(int(state.get(key, 0)) for key in (
            "executor_model_calls_used", "skill_preparation_calls_used", "worker_compaction_calls_used"))
        limit = state.get("executor_model_run_limit")
        # Reserve two summary attempts plus two normal execution/finalization calls.
        return limit is None or int(limit) - used >= 4

    def _prepare(self, state, request=None):
        if not self.enabled or not self._allowed(state):
            return None
        messages, selected = self._select(state, ignore_threshold=request is not None)
        if not selected:
            return None
        budget = None
        if request is not None:
            fixed = [request.system_message] if request.system_message is not None else []
            tools = [convert_to_openai_tool(tool) for tool in request.tools]
            response_format = request.response_format
            schema = getattr(response_format, "schema", response_format)
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            if hasattr(response_format, "schema_specs"):
                schema = [spec.json_schema for spec in response_format.schema_specs]
            elif hasattr(response_format, "schema_spec"):
                schema = response_format.schema_spec.json_schema
            envelope = json.dumps({"tools": tools, "response_format": schema, "tool_choice": request.tool_choice}, ensure_ascii=False, default=str)
            from langchain_core.messages import SystemMessage
            overhead = count_tokens_approximately([*fixed, SystemMessage(content=envelope)])
            chosen = set(selected)
            protected = overhead + count_tokens_approximately([m for i, m in enumerate(messages) if i not in chosen])
            total = overhead + count_tokens_approximately(messages)
            if total < self.threshold:
                return None
            if protected / max(1, total) >= self.settings.protected_ratio_limit:
                return None
            allowance = min(int(self.settings.target_tokens * self.settings.summary_ratio), self.settings.target_tokens - protected - self.settings.safety_tokens)
            if allowance <= 0:
                return None
            budget = {"summary_tokens": allowance, "protected_tokens": protected, "envelope": envelope}
        signature = hashlib.sha256(json.dumps(
            {"messages": [message.model_dump(mode="json") for message in messages], "budget": budget},
            ensure_ascii=False, default=str,
        ).encode()).hexdigest()
        if signature == state.get("worker_compaction_deferred_signature"):
            return None
        from workers.evidence_refs import registry
        refs = registry(state)
        return messages, selected, signature, refs, budget

    def _update(self, state, messages, selected, signature, refs, summary, attempts):
        result = self._finish(messages, selected, summary) if summary is not None else {}
        result["worker_compaction_calls_used"] = int(state.get("worker_compaction_calls_used", 0)) + attempts
        result["worker_evidence_refs"] = refs
        result["worker_archived_messages"] = [*state.get("worker_archived_messages", []), *result.pop("_archived", [])]
        result["worker_compaction_deferred_signature"] = "" if summary is not None else signature
        return result

    def before_model(self, state, runtime):
        if self.request_aware:
            return None
        return self._compact(state)

    def _compact(self, state, request=None):
        prepared = self._prepare(state, request)
        if prepared is None:
            return None
        messages, selected, signature, refs, budget = prepared
        try:
            summary, attempts = run_compaction(
                self.model, WorkerExecutionSummary, self._request(messages, selected, refs, budget),
                core_fields=("observed_results", "errors_and_failures", "pending_work"),
                max_chars=self.max_chars, check=self._check(messages, selected, refs, budget),
            )
        except CompactionFailure as error:
            summary, attempts = None, error.attempts
        return self._update(state, messages, selected, signature, refs, summary, attempts)

    async def abefore_model(self, state, runtime):
        if self.request_aware:
            return None
        return await self._acompact(state)

    async def _acompact(self, state, request=None):
        prepared = self._prepare(state, request)
        if prepared is None:
            return None
        messages, selected, signature, refs, budget = prepared
        try:
            summary, attempts = await arun_compaction(
                self.model, WorkerExecutionSummary, self._request(messages, selected, refs, budget),
                core_fields=("observed_results", "errors_and_failures", "pending_work"),
                max_chars=self.max_chars, check=self._check(messages, selected, refs, budget),
            )
        except CompactionFailure as error:
            summary, attempts = None, error.attempts
        return self._update(state, messages, selected, signature, refs, summary, attempts)

    def _forward(self, request, update):
        if update and "messages" in update:
            return request.override(messages=update["messages"][1:])
        return request

    def _response(self, response, update):
        if not update:
            return response
        # Factory applies commands after model output; include the new response
        # when replacing history, retaining summary order and tool-call state.
        update = dict(update)
        if "messages" in update:
            update["messages"] = [*update["messages"], *response.result]
        return ExtendedModelResponse(model_response=response, command=Command(update=update))

    def wrap_model_call(self, request, handler):
        if not self.request_aware:
            return handler(request)
        state = {**request.state, "messages": request.messages}
        update = self._compact(state, request)
        return self._response(handler(self._forward(request, update)), update)

    async def awrap_model_call(self, request, handler):
        if not self.request_aware:
            return await handler(request)
        state = {**request.state, "messages": request.messages}
        update = await self._acompact(state, request)
        return self._response(await handler(self._forward(request, update)), update)
