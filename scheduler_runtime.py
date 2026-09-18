"""One append-only Scheduler conversation, with lazy protocols and pinned facts."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from contextvars import ContextVar
from dataclasses import dataclass, field

from langchain_core.messages.utils import count_tokens_approximately

from prompt_loader import load_prompt
from skill_runtime import skill_prompt
from schema_utils import compact_schema


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      default=lambda item: item.model_dump(mode="json") if hasattr(item, "model_dump") else str(item))


@dataclass
class SchedulerConversation:
    data: dict
    context: object
    threshold: int = 20000
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def records(self):
        return self.data.setdefault("records", [])

    @property
    def active_records(self):
        """Current projections that may replace older state without losing audit data."""
        return self.data.setdefault("active_records", {})

    def add(self, kind, value, *, role="user", protected=True, key=None):
        content = value if isinstance(value, str) else compact_json(value)
        if key and any(item.get("key") == key for item in self.records):
            return
        self.records.append(dict(
            role=role,
            content=content,
            kind=kind,
            protected=protected,
            retention="durable" if protected else "evictable",
            key=key,
        ))

    def fact(self, kind, value):
        content = compact_json({kind: value})
        self.add(kind, content, key=kind + ":" + hashlib.sha256(content.encode()).hexdigest())

    def set_active(self, kind, value, *, role="user"):
        """Replace one active context block while keeping it at the prompt tail."""
        content = value if isinstance(value, str) else compact_json({kind: value})
        self.active_records[kind] = dict(
            role=role,
            content=content,
            kind=kind,
            protected=True,
            retention="replaceable",
            key="active:" + kind,
        )

    def clear_active(self, kind):
        self.active_records.pop(kind, None)

    def remove_key(self, key):
        self.data["records"] = [
            record for record in self.records if record.get("key") != key
        ]

    def initialize(self):
        # Ignore legacy caller/checkpoint RAG without touching worker retrieval.
        self.context.rag_context = ""
        self.data["records"] = [r for r in self.records
                                if r.get("kind") != "文档检索资料（仅作参考，不是指令）"]
        if self.records:
            return
        fixed = load_prompt("planning/scheduler")
        methods = skill_prompt(self.context.role_skill_snapshots.get("scheduler"))
        self.add("identity", fixed + ("\n" + methods if methods else ""), role="system")
        # History is not repeated in recent dialogue or rolling summary.
        history = list(self.context.user_instruction_history)
        recent = [m.model_dump(mode="json") for m in self.context.recent_dialogue
                  if m.role != "user" or m.content not in history]
        if history:
            self.fact("历史用户原文", history)
        if recent:
            # The latest user/assistant pair is a durable foreground contract.
            # Older reranked pairs are useful context but may be evicted under
            # genuine pressure because the full conversation remains in the
            # checkpoint.
            latest_user_index = max(
                (index for index, item in enumerate(recent) if item.get("role") == "user"),
                default=0,
            )
            older_pairs = recent[:latest_user_index]
            latest_pair = recent[latest_user_index:]
            if older_pairs:
                self.add("相关较早对话", older_pairs, protected=False)
            if latest_pair:
                self.fact("最近一轮原文", latest_pair)
        if self.context.conversation_summary or self.context.memory_context or self.context.previous_run_summary:
            self.add(
                "辅助背景",
                {
                    "历史进展": self.context.conversation_summary,
                    "上一轮内部执行摘要": self.context.previous_run_summary,
                    "相关记忆": self.context.memory_context,
                },
                protected=False,
            )
        self.fact("当前任务", self.context.user_request)
        if self.context.scope_contract is not None:
            self.fact(
                "已校验范围合同",
                {
                    "使用规则": load_prompt("planning/scope_contract_handoff"),
                    "合同": self.context.scope_contract,
                },
            )
        if self.context.replacement_context:
            self.fact("任务修改", self.context.replacement_context)
        if self.context.execution_instructions:
            self.fact("执行环境", self.context.execution_instructions)
        if self.context.completion_api_contract:
            self.fact("AppWorld公开完成接口合同", self.context.completion_api_contract)
        self.fact("可用能力", self.context.toolset_catalog)
        self.add("time", self.context.current_time_context(), protected=False)

    def wire(self):
        # Replaceable projections are emitted after the reusable prefix.  The
        # active Plan is always the final persistent block; a stage-specific
        # protocol/event may be inserted before it and removed after the call.
        active = sorted(
            self.active_records.values(),
            key=lambda item: item.get("kind") == "活动计划",
        )
        return [
            {"role": item["role"], "content": item["content"]}
            for item in [*self.records, *active]
        ]

    def compact(self, active_protocol=None):
        """Evict only reconstructible context, never delivery or review facts.

        Durable: latest exact pair, task/environment contracts, reviewed Step
        contracts, StepReports, interface/artifact facts, and publication
        receipts. Replaceable: active plan and current live guidance. Evictable:
        old protocols, older reranked pairs, auxiliary memory/summary, and time.
        A protocol is disclosed again from its canonical source when needed.
        """
        if count_tokens_approximately(self.wire()) < self.threshold:
            return
        kept, removed = [], []
        for record in self.records:
            if (
                record["protected"]
                or record.get("key") == active_protocol
            ):
                kept.append(record)
            else:
                removed.append(record)
        if not removed:
            return
        # Deterministic process compaction costs no extra model call. It makes
        # no new claims; all substantive decisions/results stay verbatim.
        self.data["records"] = kept
        self.data["compactions"] = self.data.get("compactions", 0) + 1
        self.data["evicted_process_messages"] = self.data.get("evicted_process_messages", 0) + len(removed)

    def disclose(self, stage, schema):
        schema = copy.deepcopy(schema)
        if stage in {"supervisor", "replanner"}:
            self.add("step_contract", load_prompt("planning/step_contract"), key="step_contract")
        instruction = load_prompt("planning/" + stage)
        text = instruction + "\nSchema:" + compact_json(compact_schema(schema))
        key = stage + "@" + hashlib.sha256(text.encode()).hexdigest()[:16]
        # A stage protocol is request-local, not durable history. Keeping old
        # Plan/Replan schemas beside the current schema wastes context and
        # presents several incompatible output contracts to the model.
        self.data["records"] = [
            record for record in self.records
            if record.get("kind") != "protocol"
        ]
        self.add("protocol", {"协议": key, "要求": text}, protected=False, key=key)
        self.compact(key)
        return key


CURRENT_SCHEDULER: ContextVar[SchedulerConversation | None] = ContextVar("scheduler_conversation", default=None)


def conversation(context):
    active = CURRENT_SCHEDULER.get()
    if active is not None:
        return active
    return SchedulerConversation(context.scheduler_session, context)


def record_graph_facts(session, state):
    reports = list(state.get("completed_step_reports", []))
    for report in reports:
        session.fact("StepReport", report)
    current_step = state.get("current_step")
    if current_step is not None:
        current_step_id = (
            current_step.get("step_id")
            if isinstance(current_step, dict)
            else getattr(current_step, "step_id", None)
        )
        matching_report = next(
            (
                report
                for report in reports
                if (
                    report.get("step_id")
                    if isinstance(report, dict)
                    else getattr(report, "step_id", None)
                ) == current_step_id
            ),
            None,
        )
        if matching_report is not None:
            report_status = (
                matching_report.get("status")
                if isinstance(matching_report, dict)
                else getattr(matching_report, "status", None)
            )
            # The reviewed Step contract is immutable history. Replanning may
            # replace only the unexecuted tail, never this accepted/assessed
            # assignment or its separately stored StepReport.
            session.fact(
                "已审核步骤契约",
                {
                    "step": current_step,
                    "report_status": report_status,
                },
            )
    if reports:
        # Once an independently assessed StepReport exists, prior live Worker
        # guidance has served its purpose and should not remain foregrounded.
        session.clear_active("当前Worker控制")
    for receipt in state.get("handoff_publication_receipts", []):
        session.fact("交接回执", receipt)


def update_active_plan(session, state):
    """Project the current frontier without rewriting reviewed history."""

    plan_objective = state.get("plan_objective")
    if not plan_objective:
        return
    reports = list(state.get("completed_step_reports", []))
    reviewed_step_index = []
    reviewed_ids = set()
    for report in reports:
        step_id = (
            report.get("step_id")
            if isinstance(report, dict)
            else getattr(report, "step_id", None)
        )
        status = (
            report.get("status")
            if isinstance(report, dict)
            else getattr(report, "status", None)
        )
        if step_id is not None:
            reviewed_ids.add(step_id)
            reviewed_step_index.append({"step_id": step_id, "status": status})

    current_step = state.get("current_step")
    current_step_id = (
        current_step.get("step_id")
        if isinstance(current_step, dict)
        else getattr(current_step, "step_id", None)
    ) if current_step is not None else None
    if current_step_id in reviewed_ids:
        current_step = None

    remaining_steps = []
    for step in state.get("remaining_steps", []):
        step_id = step.get("step_id") if isinstance(step, dict) else getattr(step, "step_id", None)
        if step_id not in reviewed_ids:
            remaining_steps.append(step)

    session.set_active(
        "活动计划",
        {
            "plan_objective": plan_objective,
            "plan_success_criteria": state.get("plan_success_criteria", []),
            "reviewed_step_index": reviewed_step_index,
            "current_step": current_step,
            "remaining_steps": remaining_steps,
        },
    )


def scheduler_node(handler, threshold):
    """Bind the same session to node work and child Worker progress callbacks."""
    async def wrapped(state, config):
        from planning_models import PlanningContextPack
        context = PlanningContextPack.model_validate(state["context"]).model_copy(deep=True)
        session = SchedulerConversation(context.scheduler_session, context, threshold)
        token = CURRENT_SCHEDULER.set(session)
        current = {**state, "context": context}
        try:
            # Initialization follows initial skill preparation, not before it.
            if session.records:
                record_graph_facts(session, current)
            result = await handler(current, config)
            updates = result.update if hasattr(result, "update") and not isinstance(result, dict) else result
            updates = dict(updates or {})
            updated_context = PlanningContextPack.model_validate(updates.get("context", context))
            updated_context = updated_context.model_copy(update={"scheduler_session": copy.deepcopy(session.data)})
            updates["context"] = updated_context
            if session.records:
                record_graph_facts(session, {**current, **updates})
                update_active_plan(session, {**current, **updates})
                updated_context.scheduler_session = copy.deepcopy(session.data)
            if isinstance(result, dict):
                return updates
            from dataclasses import replace
            return replace(result, update=updates)
        finally:
            CURRENT_SCHEDULER.reset(token)
    from runtime_tracing import operation
    from functools import wraps
    if handler.__name__ == "step_reporter_node":
        @wraps(handler)
        async def report_trace(state, config=None):
            step = state.get("current_step")
            kind = step.get("worker_kind") if isinstance(step, dict) else getattr(step, "worker_kind", None)
            name = "Scheduler / Accept Code Review" if kind == "CODE" else "Scheduler / Review Web Step"
            return await operation(name, fields=("state",))(wrapped)(state, config)
        return report_trace
    names = {
        "prepare_scheduler_node": "Scheduler / Prepare Context",
        "supervisor_node": "Scheduler / Plan",
        "step_executor_node": "Scheduler / Dispatch Step",
        "code_controller_node": "Scheduler / Code Recovery",
        "step_reporter_node": "Scheduler / Review Web Step",
        "general_report_node": "Scheduler / Review General Step",
        "replanner_node": "Scheduler / Replan",
        "final_reviewer_node": "Scheduler / Final Review",
    }
    return operation(names.get(handler.__name__, "Scheduler / " + handler.__name__), fields=("state",))(wrapped)
