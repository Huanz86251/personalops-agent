"""Offline integration tests for durable Worker lifecycle coordination."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from eventing import AsyncEventStore
from planning_models import PlanStep, WorkerAssignment
from workers.coordinator import WorkerGroupCoordinator
from workers.group import (
    StepWorkerGroup,
    WorkerAttemptStatus,
    WorkerGroupReviewStatus,
    WorkerWorkspaceStatus,
)


T0 = datetime(2026, 9, 3, 11, 0, tzinfo=timezone.utc)


def step() -> PlanStep:
    return PlanStep(
        step_id=1,
        objective="Research two sources.",
        success_criteria=["Both source directions are handled."],
        worker_kind="WEB",
        execution_mode="PARALLEL",
        worker_assignments=[
            WorkerAssignment(assignment_key="first", objective="First source."),
            WorkerAssignment(assignment_key="second", objective="Second source."),
        ],
    )


def current(group: StepWorkerGroup, key: str):
    return next(
        slot.current_attempt for slot in group.slots
        if slot.assignment_key == key
    )


class WorkerGroupCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_create_replays_without_new_timestamp(self):
        first = await self.coordinator.create_group(
            step(),
            event_id="event-coordinator",
            group_id="stable-group",
            created_at=T0,
        )
        replay = await self.coordinator.create_group(
            step(),
            event_id="event-coordinator",
            group_id="stable-group",
            created_at=T0 + timedelta(minutes=5),
        )

        self.assertEqual(replay, first)

    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "events.sqlite3"
        self.store = AsyncEventStore(self.path)
        await self.store.start()
        self.coordinator = WorkerGroupCoordinator(
            self.store,
            workspace_retention_minutes=30,
        )

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.temporary.cleanup()

    async def create_running_group(self) -> StepWorkerGroup:
        stored = await self.coordinator.create_group(
            step(),
            event_id="event-coordinator",
            group_id="group-coordinator",
            created_at=T0,
        )
        group = StepWorkerGroup.model_validate(stored.snapshot)
        for offset, key in enumerate(("first", "second"), 1):
            attempt = current(group, key)
            result = await self.coordinator.start_attempt(
                group_id=group.group_id,
                assignment_key=key,
                attempt_no=attempt.attempt_no,
                worker_id=attempt.worker_id,
                occurred_at=T0 + timedelta(seconds=offset),
            )
            group = result.group
        return group

    async def test_start_and_finish_persist_state_and_audit_together(self):
        group = await self.create_running_group()
        attempt = current(group, "first")
        result = await self.coordinator.finish_attempt(
            group_id=group.group_id,
            assignment_key="first",
            attempt_no=attempt.attempt_no,
            worker_id=attempt.worker_id,
            status=WorkerAttemptStatus.SUBMITTED,
            terminal_reason="First source submitted.",
            occurred_at=T0 + timedelta(seconds=3),
            trace_id="trace-first",
        )

        self.assertTrue(result.applied)
        self.assertEqual(
            current(result.group, "first").workspace.status,
            WorkerWorkspaceStatus.FROZEN,
        )
        self.assertEqual(
            current(result.group, "first").workspace.retain_until,
            T0 + timedelta(minutes=30, seconds=3),
        )
        audits = await self.store.list_worker_attempt_audit(
            group_id=group.group_id,
            assignment_key="first",
        )
        self.assertEqual([item["operation"] for item in audits], ["START", "FINISH"])
        self.assertEqual(audits[-1]["trace_id"], "trace-first")

    async def test_concurrent_finishes_retry_cas_and_merge_both_slots(self):
        group = await self.create_running_group()
        other_store = AsyncEventStore(self.path)
        await other_store.start()
        other = WorkerGroupCoordinator(other_store, workspace_retention_minutes=30)
        first = current(group, "first")
        second = current(group, "second")
        try:
            results = await asyncio.gather(
                self.coordinator.finish_attempt(
                    group_id=group.group_id,
                    assignment_key="first",
                    attempt_no=first.attempt_no,
                    worker_id=first.worker_id,
                    status=WorkerAttemptStatus.SUBMITTED,
                    terminal_reason="First complete.",
                    occurred_at=T0 + timedelta(seconds=3),
                ),
                other.finish_attempt(
                    group_id=group.group_id,
                    assignment_key="second",
                    attempt_no=second.attempt_no,
                    worker_id=second.worker_id,
                    status=WorkerAttemptStatus.SUBMITTED,
                    terminal_reason="Second complete.",
                    occurred_at=T0 + timedelta(seconds=3),
                ),
            )
            self.assertTrue(all(result.applied for result in results))
            stored = await self.store.require_worker_group(group.group_id)
            merged = StepWorkerGroup.model_validate(stored.snapshot)
            self.assertEqual(
                merged.review_status,
                WorkerGroupReviewStatus.JOIN_READY,
            )
            self.assertEqual(
                {attempt.status for attempt in merged.current_attempts},
                {WorkerAttemptStatus.SUBMITTED},
            )
        finally:
            await other_store.close()

    async def test_replace_then_late_old_callback_is_audited_as_stale(self):
        group = await self.create_running_group()
        old = current(group, "first")
        replaced = await self.coordinator.replace_attempt(
            group_id=group.group_id,
            assignment_key="first",
            attempt_no=old.attempt_no,
            worker_id=old.worker_id,
            replacement_objective="Use the corrected official query.",
            reason="Original query targeted the wrong location.",
            occurred_at=T0 + timedelta(seconds=3),
        )
        late = await self.coordinator.finish_attempt(
            group_id=group.group_id,
            assignment_key="first",
            attempt_no=old.attempt_no,
            worker_id=old.worker_id,
            status=WorkerAttemptStatus.SUBMITTED,
            terminal_reason="Late obsolete result.",
            occurred_at=T0 + timedelta(seconds=4),
            trace_id="trace-stale",
        )

        self.assertTrue(replaced.applied)
        self.assertFalse(late.applied)
        self.assertTrue(late.stale)
        latest = current(late.group, "first")
        self.assertEqual(latest.attempt_no, 2)
        self.assertEqual(latest.workspace.status, WorkerWorkspaceStatus.PREPARED)
        audits = await self.store.list_worker_attempt_audit(
            group_id=group.group_id,
            assignment_key="first",
        )
        self.assertEqual(audits[-1]["operation"], "CALLBACK")
        self.assertEqual(audits[-1]["outcome"], "STALE")
        self.assertEqual(audits[-1]["trace_id"], "trace-stale")


if __name__ == "__main__":
    unittest.main()
