"""Control-plane bridge between Deep Agent Workers and the main planner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from langgraph.types import Command

from eventing.store import AsyncEventStore
from workers.leadership_models import (
    LeadershipDecision,
    LeadershipDecisionResult,
    LeadershipWakeRequest,
    LeadershipWorkerView,
)
from workers.progress import WorkerProgressRecord
from workers.wake_policy import LeadershipWakePolicy


PLANNING_LEADER_ID = "planning_supervisor"
LeadershipDecisionHandler = Callable[
    [LeadershipWakeRequest],
    Awaitable[LeadershipDecisionResult],
]


@dataclass
class LeadershipCoordinationState:
    """Shared active-Worker state for bridges wrapping different graphs.

    Specialized runtimes may build one graph per isolated resource lease.  The
    graph instances must remain separate, while their progress gates still
    need one event-level view so a multi-Worker wake is coalesced correctly.
    """

    active_workers: dict[str, dict[str, LeadershipWorkerView]] = field(
        default_factory=dict
    )
    event_locks: dict[str, asyncio.Lock] = field(default_factory=dict)


class LeadershipProgressBatch(BaseModel):
    """Unread Worker evidence presented to one leadership consumer."""

    consumer_id: str
    worker_id: str
    cursor_before: int = Field(ge=0)
    latest_sequence: int = Field(ge=0)
    reports: list[WorkerProgressRecord]


class WorkerLeadershipBridge:
    """Run one Worker while durably collecting its custom progress stream.

    The bridge is deliberately policy-free. It makes progress visible and
    checkpointed, but it does not decide whether a report should wake a model,
    send feedback, cancel a Worker, or replace it. Those are leadership policy
    decisions layered on top of this transport boundary.
    """

    def __init__(
        self,
        worker_graph,
        event_store: AsyncEventStore,
        *,
        wake_policy: LeadershipWakePolicy | None = None,
        decision_handler: LeadershipDecisionHandler | None = None,
        coordination_state: LeadershipCoordinationState | None = None,
    ) -> None:
        self.worker_graph = worker_graph
        self.event_store = event_store
        self.wake_policy = wake_policy or LeadershipWakePolicy()
        self.decision_handler = decision_handler
        self.coordination_state = (
            coordination_state or LeadershipCoordinationState()
        )
        self._active_workers = self.coordination_state.active_workers
        self._event_locks = self.coordination_state.event_locks

    @staticmethod
    def _assignment_from_input(input_state: Any) -> str:
        if not isinstance(input_state, dict):
            return "No assignment text was available."
        messages = list(input_state.get("messages", []))
        if not messages:
            return "No assignment text was available."
        message = messages[-1]
        content = (
            message.get("content", "")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        normalized = str(content).strip()
        return normalized or "No assignment text was available."

    async def register_worker(
        self,
        *,
        event_id: str,
        worker_id: str,
        step_id: str | None,
        assignment: str,
    ) -> None:
        lock = self._event_locks.setdefault(event_id, asyncio.Lock())
        async with lock:
            self._active_workers.setdefault(event_id, {})[worker_id] = (
                LeadershipWorkerView(
                    worker_id=worker_id,
                    step_id=step_id,
                    assignment=assignment,
                    cursor_before=0,
                    reports=[],
                )
            )

    async def unregister_worker(self, event_id: str, worker_id: str) -> None:
        lock = self._event_locks.setdefault(event_id, asyncio.Lock())
        async with lock:
            workers = self._active_workers.get(event_id)
            if workers is None:
                return
            workers.pop(worker_id, None)
            if not workers:
                self._active_workers.pop(event_id, None)

    @staticmethod
    def _auto_continue(reason: str, *, model_rounds_used: int = 0):
        return LeadershipDecisionResult(
            decision=LeadershipDecision(
                action="CONTINUE",
                reason=reason,
            ),
            model_rounds_used=model_rounds_used,
        )

    async def decide_at_progress_gate(
        self,
        record: WorkerProgressRecord,
    ) -> LeadershipDecisionResult:
        """Coalesce unread reports and invoke the leader at most once per event."""

        worker_id = record.worker_id or ""
        event_id = record.event_id or ""
        pending = await self.event_store.get_pending_worker_directive(worker_id)
        if pending is not None:
            return LeadershipDecisionResult.model_validate(pending["directive"])

        lock = self._event_locks.setdefault(event_id, asyncio.Lock())
        async with lock:
            pending = await self.event_store.get_pending_worker_directive(worker_id)
            if pending is not None:
                return LeadershipDecisionResult.model_validate(
                    pending["directive"]
                )

            active = dict(self._active_workers.get(event_id, {}))
            if worker_id not in active:
                return self._auto_continue(
                    "Worker was no longer active when its checkpoint was evaluated."
                )

            batches: dict[str, LeadershipProgressBatch] = {}
            reports_by_worker: dict[str, list[WorkerProgressRecord]] = {}
            for active_worker_id in sorted(active):
                batch = await self.read_pending_progress(
                    worker_id=active_worker_id
                )
                batches[active_worker_id] = batch
                reports_by_worker[active_worker_id] = batch.reports

            wake_reason = self.wake_policy.evaluate(
                active_worker_ids=tuple(sorted(active)),
                reports_by_worker=reports_by_worker,
            )
            if wake_reason is None:
                return self._auto_continue(
                    "Deterministic wake threshold has not been reached."
                )
            if self.decision_handler is None:
                return self._auto_continue(
                    "No leadership model handler is configured."
                )

            wake_request = LeadershipWakeRequest(
                wake_id=f"wake_{uuid4().hex}",
                event_id=event_id,
                reason=wake_reason,
                created_at=datetime.now(timezone.utc),
                workers=[
                    active[active_worker_id].model_copy(
                        update={
                            "cursor_before": batches[active_worker_id].cursor_before,
                            "reports": [
                                report.model_dump(mode="json")
                                for report in reports_by_worker[active_worker_id]
                            ],
                        }
                    )
                    for active_worker_id in sorted(active)
                ],
            )
            if wake_reason == "WORKER_POSSIBLY_READY":
                # Readiness is not a planning question.  Stop further Web
                # exploration deterministically and let the independent Step
                # Reporter assess the submitted evidence.  The Scheduler is
                # reserved for genuine control decisions such as a blocked
                # path, replacement, or later replan.
                result = LeadershipDecisionResult(
                    decision=LeadershipDecision(
                        action="ACCEPT",
                        reason=(
                            "Worker declared its evidence ready for independent "
                            "review; no Scheduler model call was needed."
                        ),
                    ),
                    model_rounds_used=0,
                )
            else:
                result = await self.decision_handler(wake_request)
            decision = result.decision
            if decision.target_worker_ids:
                target_ids = list(decision.target_worker_ids)
            elif decision.action in {"ACCEPT", "CANCEL"}:
                target_ids = list(sorted(active))
            else:
                target_ids = [worker_id]

            unknown_targets = sorted(set(target_ids).difference(active))
            if unknown_targets:
                result = LeadershipDecisionResult(
                    decision=LeadershipDecision(
                        action="CONTINUE",
                        reason=(
                            "Leadership selected unknown active worker IDs; "
                            f"safe fallback used: {unknown_targets}"
                        ),
                    ),
                    model_rounds_used=result.model_rounds_used,
                    used_fallback=True,
                )
                target_ids = [worker_id]

            directives: dict[str, LeadershipDecisionResult] = {}
            for target_id in target_ids:
                targeted_decision = result.decision.model_copy(
                    update={"target_worker_ids": [target_id]}
                )
                directives[target_id] = LeadershipDecisionResult(
                    decision=targeted_decision,
                    model_rounds_used=(
                        result.model_rounds_used if target_id == worker_id else 0
                    ),
                    used_fallback=result.used_fallback,
                    wake_id=wake_request.wake_id,
                )

            if worker_id not in directives:
                directives[worker_id] = self._auto_continue(
                    "Leadership action targeted another Worker.",
                    model_rounds_used=result.model_rounds_used,
                ).model_copy(update={"wake_id": wake_request.wake_id})

            await self.event_store.save_leadership_wake(
                wake=wake_request.model_dump(mode="json"),
                result=result.model_dump(mode="json"),
                directives={
                    target_id: directive.model_dump(mode="json")
                    for target_id, directive in directives.items()
                },
                cursor_updates={
                    active_worker_id: batch.latest_sequence
                    for active_worker_id, batch in batches.items()
                },
                consumer_id=PLANNING_LEADER_ID,
            )
            return directives[worker_id]

    async def ingest_custom_chunk(self, chunk: Any) -> WorkerProgressRecord | None:
        """Validate and persist one recognized Worker custom-stream chunk."""

        if not isinstance(chunk, dict) or chunk.get("type") != "worker_progress":
            return None

        record = WorkerProgressRecord.model_validate(chunk.get("record"))
        if not record.worker_id:
            raise ValueError("Worker progress cannot reach leadership without worker_id.")
        if not record.event_id:
            raise ValueError("Worker progress cannot reach leadership without event_id.")

        await self.event_store.append_worker_progress(
            record.model_dump(mode="json")
        )
        return record

    async def ainvoke(
        self,
        input_state,
        config=None,
        **kwargs,
    ):
        """Invoke the Worker and consume progress before returning final state."""

        if "stream_mode" in kwargs:
            raise ValueError(
                "WorkerLeadershipBridge owns stream_mode so progress cannot be bypassed."
            )

        resolved_input = input_state
        event_id = ""
        worker_id = ""
        step_id = None
        control_enabled = self.decision_handler is not None
        configurable = (config or {}).get("configurable", {})
        event_pause_control = configurable.get("event_pause_control")
        resume_from_checkpoint = bool(
            configurable.get("resume_from_checkpoint")
        )
        if isinstance(input_state, dict):
            event_id = str(input_state.get("event_id") or "").strip()
            worker_id = str(input_state.get("worker_id") or "").strip()
            raw_step_id = input_state.get("step_id")
            step_id = None if raw_step_id is None else str(raw_step_id)
            resolved_input = dict(input_state)
            resolved_input["worker_control_enabled"] = control_enabled

        if control_enabled and (not event_id or not worker_id):
            raise ValueError(
                "Controlled Worker invocation requires event_id and worker_id."
            )
        if control_enabled and not (
            isinstance(config, dict)
            and isinstance(config.get("configurable"), dict)
            and config["configurable"].get("thread_id")
        ):
            raise ValueError("Controlled Worker invocation requires thread_id.")

        if event_id and worker_id:
            await self.register_worker(
                event_id=event_id,
                worker_id=worker_id,
                step_id=step_id,
                assignment=self._assignment_from_input(input_state),
            )

        resume_worker_from_checkpoint = False
        if resume_from_checkpoint:
            checkpoint_state = await self.worker_graph.aget_state(config)
            resume_worker_from_checkpoint = bool(
                getattr(checkpoint_state, "created_at", None)
                or getattr(checkpoint_state, "values", None)
                or getattr(checkpoint_state, "next", ())
            )
            if resume_worker_from_checkpoint:
                resolved_input = None

        try:
            delivery_wake_id: str | None = None
            while True:
                final_values = None
                latest_progress: WorkerProgressRecord | None = None
                async for stream_item in self.worker_graph.astream(
                    resolved_input,
                    config=config,
                    stream_mode=["custom", "values"],
                    **kwargs,
                ):
                    if not isinstance(stream_item, tuple) or len(stream_item) != 2:
                        raise RuntimeError(
                            "LangGraph returned an unsupported multi-mode stream item."
                        )

                    mode, payload = stream_item
                    if mode == "custom":
                        ingested = await self.ingest_custom_chunk(payload)
                        if ingested is not None:
                            latest_progress = ingested
                    elif mode == "values":
                        final_values = payload
                        if event_pause_control is not None:
                            # General/Web expose their existing LangGraph node
                            # boundaries to top-level INSERT. The Agent graph
                            # and any Playwright lease remain alive while the
                            # urgent event is handled.
                            await event_pause_control.pause_point()
                        if delivery_wake_id and isinstance(payload, dict):
                            applied = any(
                                isinstance(item, dict)
                                and item.get("wake_id") == delivery_wake_id
                                for item in payload.get(
                                    "worker_leadership_decisions",
                                    [],
                                )
                            )
                            if applied:
                                await self.event_store.mark_worker_directive_applied(
                                    wake_id=delivery_wake_id,
                                    worker_id=worker_id,
                                )
                                delivery_wake_id = None

                if final_values is None and resume_worker_from_checkpoint:
                    completed_state = await self.worker_graph.aget_state(config)
                    completed_values = (
                        getattr(completed_state, "values", {}) or {}
                    )
                    if isinstance(completed_values, dict):
                        final_values = dict(completed_values)

                if not control_enabled:
                    if final_values is None:
                        raise RuntimeError(
                            "Worker completed without a final values snapshot."
                        )
                    return final_values

                snapshot = await self.worker_graph.aget_state(config)
                interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
                if interrupts:
                    gate = getattr(interrupts[0], "value", None)
                    if not isinstance(gate, dict) or gate.get("type") != (
                        "worker_progress_gate"
                    ):
                        raise RuntimeError("Worker stopped at an unknown interrupt.")
                    gate_record = WorkerProgressRecord.model_validate(
                        gate.get("record")
                    )
                    if latest_progress is None:
                        await self.event_store.append_worker_progress(
                            gate_record.model_dump(mode="json")
                        )
                    decision = await self.decide_at_progress_gate(gate_record)
                    delivery_wake_id = decision.wake_id
                    resolved_input = Command(
                        resume=decision.model_dump(mode="json")
                    )
                    continue

                if final_values is None:
                    raise RuntimeError(
                        "Worker completed without a final values snapshot."
                    )
                return final_values
        finally:
            if event_id and worker_id:
                await self.unregister_worker(event_id, worker_id)

    async def aget_state(self, *args, **kwargs):
        """Delegate checkpoint reads to the wrapped Deep Agent graph."""

        return await self.worker_graph.aget_state(*args, **kwargs)

    async def aupdate_state(self, *args, **kwargs):
        """Delegate metadata writes to the wrapped Deep Agent graph."""

        return await self.worker_graph.aupdate_state(*args, **kwargs)

    async def read_pending_progress(
        self,
        *,
        worker_id: str,
        consumer_id: str = PLANNING_LEADER_ID,
        limit: int = 100,
    ) -> LeadershipProgressBatch:
        """Read reports after the leader's durable cursor without acknowledging."""

        cursor = await self.event_store.get_worker_progress_cursor(
            consumer_id=consumer_id,
            worker_id=worker_id,
        )
        raw_reports = await self.event_store.list_worker_progress(
            worker_id=worker_id,
            after_sequence=cursor,
            limit=limit,
        )
        reports = [
            WorkerProgressRecord.model_validate(record)
            for record in raw_reports
        ]
        latest_sequence = reports[-1].sequence if reports else cursor
        return LeadershipProgressBatch(
            consumer_id=consumer_id,
            worker_id=worker_id,
            cursor_before=cursor,
            latest_sequence=latest_sequence,
            reports=reports,
        )

    async def acknowledge_progress(
        self,
        batch: LeadershipProgressBatch,
    ) -> int:
        """Commit that one leadership consumer processed this exact batch."""

        return await self.event_store.advance_worker_progress_cursor(
            consumer_id=batch.consumer_id,
            worker_id=batch.worker_id,
            sequence=batch.latest_sequence,
        )

    async def read_event_progress(
        self,
        event_id: str,
        *,
        limit: int = 100,
    ) -> list[WorkerProgressRecord]:
        """Build a cross-Worker event snapshot for the outer planning layer."""

        records = await self.event_store.list_event_worker_progress(
            event_id,
            limit=limit,
        )
        return [WorkerProgressRecord.model_validate(record) for record in records]


__all__ = [
    "LeadershipCoordinationState",
    "LeadershipProgressBatch",
    "PLANNING_LEADER_ID",
    "WorkerLeadershipBridge",
]
