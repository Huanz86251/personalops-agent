"""Compact old execution chatter, preserving assignments and control handoffs."""
import json
import os

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from prompt_loader import load_prompt


CONTROL = {"submit_code_for_review", "respond_to_code_review", "submit_continued_code_for_review",
           "request_code_worker_repair", "publish_reviewed_candidate", "submit_code_review",
           "submit_for_review", "report_general_result"}


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
    ):
        if isinstance(model, str):
            from langchain.chat_models import init_chat_model
            model = init_chat_model(model)
        self.model = model
        self.enabled = (
            bool(enabled)
            if enabled is not None
            else os.getenv("WORKER_COMPACTION_ENABLED", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.threshold = threshold or int(
            os.getenv("WORKER_SUMMARY_TRIGGER_TOKENS", "10000")
        )
        self.keep_messages = (
            int(keep_messages)
            if keep_messages is not None
            else int(os.getenv("WORKER_SUMMARY_KEEP_MESSAGES", "2"))
        )
        self.keep_tokens = (
            int(keep_tokens)
            if keep_tokens is not None
            else int(os.getenv("WORKER_SUMMARY_KEEP_TOKENS", "4000"))
        )
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

    def _select(self, state):
        messages = list(state.get("messages", []))
        if count_tokens_approximately(messages) < self.threshold:
            return messages, []
        # Keep a recent 4k-token tail (at least two messages), every
        # original/runtime instruction, and complete control exchanges.
        boundary = self._recent_boundary(messages)
        selected = []
        index = 0
        while index < boundary:
            message = messages[index]
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

    def _request(self, messages, selected, refs=None):
        from workers.evidence_refs import display
        return [{"role": "system", "content": load_prompt("conversation/worker_summary")},
                {"role": "user", "content": json.dumps(
                    display([messages[index].model_dump(mode="json") for index in selected], refs or {}),
                    ensure_ascii=False, separators=(",", ":"))}]

    def _finish(self, messages, selected, response):
        text = getattr(response, "content", "")
        if not isinstance(text, str) or not text.strip():
            return None
        # Do not accept an expansive summary that saves no context.
        summary = HumanMessage(content="执行过程摘要：\n" + text,
                               additional_kwargs={"personalops_runtime_event": True})
        if count_tokens_approximately([summary]) >= count_tokens_approximately([messages[i] for i in selected]):
            return None
        chosen = set(selected)
        retained = []
        for index, message in enumerate(messages):
            if index == selected[0]:
                retained.append(summary)
            if index not in chosen:
                retained.append(message)
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *retained],
                "_archived": [messages[index] for index in selected],
                "worker_compaction_calls_used": 1}

    def _allowed(self, state):
        used = int(state.get("executor_model_calls_used", 0)) + int(state.get("skill_preparation_calls_used", 0))
        used += int(state.get("worker_compaction_calls_used", 0))
        limit = state.get("executor_model_run_limit")
        return limit is None or int(limit) - used >= 3

    def before_model(self, state, runtime):
        if not self.enabled:
            return None
        if not self._allowed(state):
            return None
        messages, selected = self._select(state)
        if not selected:
            return None
        try:
            from workers.evidence_refs import registry
            refs = registry(state)
            response = self.model.invoke(self._request(messages, selected, refs))
            result = self._finish(messages, selected, response) or {}
        except Exception:
            result = {}
        result["worker_compaction_calls_used"] = int(state.get("worker_compaction_calls_used", 0)) + 1
        result['worker_evidence_refs'] = registry(state)
        result["worker_archived_messages"] = [*state.get("worker_archived_messages", []), *result.pop("_archived", [])]
        return result

    async def abefore_model(self, state, runtime):
        if not self.enabled:
            return None
        if not self._allowed(state):
            return None
        messages, selected = self._select(state)
        if not selected:
            return None
        try:
            from workers.evidence_refs import registry
            refs = registry(state)
            response = await self.model.ainvoke(self._request(messages, selected, refs))
            result = self._finish(messages, selected, response) or {}
        except Exception:
            result = {}
        result["worker_compaction_calls_used"] = int(state.get("worker_compaction_calls_used", 0)) + 1
        result['worker_evidence_refs'] = registry(state)
        result["worker_archived_messages"] = [*state.get("worker_archived_messages", []), *result.pop("_archived", [])]
        return result
