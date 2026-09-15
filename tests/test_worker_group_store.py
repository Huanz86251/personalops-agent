"""Offline SQLite concurrency tests for durable Worker Groups."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from eventing import AsyncEventStore, EventConflictError
from planning_models import PlanStep, WorkerAssignment
from workers.group import (
    StepWorkerGroup,
    WorkerAttemptStatus,
    WorkerGroupReviewStatus,
    create_step_worker_group,
    finish_worker_attempt,
    start_worker_attempt,
)


T0 = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)


def make_group() -> StepWorkerGroup:
    step = PlanStep(
        step_id=1,
        objective="Research two independent sources.",
        success_criteria=["Return both source findings."],
        worker_kind="WEB",
        execution_mode="PARALLEL",
        worker_assignments=[
            WorkerAssignment(assignment_key="first", objective="First source."),
            WorkerAssignment(assignment_key="second", objective="Second source."),
        ],
    )
    group = create_step_worker_group(
        step,
        event_id="event-store",
        group_id="group-store",
        created_at=T0,
    )
    for seconds, key in enumerate(("first", "second"), 1):
        attempt = next(
            slot.current_attempt for slot in group.slots
            if slot.assignment_key == key
        )
        group = start_worker_attempt(
            group,
            assignment_key=key,
            attempt_no=attempt.attempt_no,
            worker_id=attempt.worker_id,
            changed_at=T0 + timedelta(seconds=seconds),
        ).group
    return group


def finish_slot(
    group: StepWorkerGroup,
    key: str,
    seconds: int,
) -> StepWorkerGroup:
    attempt = next(
        slot.current_attempt for slot in group.slots
        if slot.assignment_key == key
    )
    return finish_worker_attempt(
        group,
        assignment_key=key,
        attempt_no=attempt.attempt_no,
        worker_id=attempt.worker_id,
        status=WorkerAttemptStatus.SUBMITTED,
        terminal_reason=f"{key} submitted.",
        changed_at=T0 + timedelta(seconds=seconds),
    ).group


class WorkerGroupStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "events.sqlite3"
        self.store = AsyncEventStore(self.path)
        await self.store.start()

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.temporary.cleanup()

    async def test_group_snapshot_persists_with_revision_and_reopens(self):
        group = make_group()
        created = await self.store.create_worker_group(
            group.model_dump(mode="json")
        )
        self.assertEqual(created.revision, 1)
        self.assertEqual(
            StepWorkerGroup.model_validate(created.snapshot),
            group,
        )

        await self.store.close()
        await self.store.start()
        reopened = await self.store.require_worker_group(group.group_id)
        self.assertEqual(reopened, created)

    async def test_stale_cas_cannot_overwrite_another_worker_completion(self):
        group = make_group()
        created = await self.store.create_worker_group(
            group.model_dump(mode="json")
        )
        first_update = finish_slot(group, "first", 3)
        second_stale_update = finish_slot(group, "second", 4)

        first_write = await self.store.compare_and_set_worker_group(
            first_update.model_dump(mode="json"),
            expected_revision=created.revision,
        )
        self.assertTrue(first_write.applied)
        stale_write = await self.store.compare_and_set_worker_group(
            second_stale_update.model_dump(mode="json"),
            expected_revision=created.revision,
        )
        self.assertFalse(stale_write.applied)

        latest = StepWorkerGroup.model_validate(stale_write.record.snapshot)
        merged_update = finish_slot(latest, "second", 4)
        merged_write = await self.store.compare_and_set_worker_group(
            merged_update.model_dump(mode="json"),
            expected_revision=stale_write.record.revision,
        )
        self.assertTrue(merged_write.applied)
        self.assertEqual(
            merged_write.record.snapshot["review_status"],
            WorkerGroupReviewStatus.JOIN_READY.value,
        )

    async def test_audit_conflict_rolls_back_group_snapshot_update(self):
        group = make_group()
        created = await self.store.create_worker_group(
            group.model_dump(mode="json")
        )
        attempt = next(
            slot.current_attempt for slot in group.slots
            if slot.assignment_key == "first"
        )
        audit = {
            "audit_id": "audit-transaction-boundary",
            "group_id": group.group_id,
            "assignment_key": "first",
            "attempt_no": attempt.attempt_no,
            "worker_id": attempt.worker_id,
            "workspace_id": attempt.workspace.workspace_id,
            "checkpoint_thread_id": attempt.workspace.checkpoint_thread_id,
            "operation": "FINISH",
            "outcome": "APPLIED",
            "occurred_at": (T0 + timedelta(seconds=3)).isoformat(),
            "recorded_at": (T0 + timedelta(seconds=3)).isoformat(),
            "summary": "Original audit row.",
            "trace_id": None,
            "status_before": WorkerAttemptStatus.RUNNING.value,
            "status_after": WorkerAttemptStatus.SUBMITTED.value,
        }
        await self.store.append_worker_attempt_audit(audit)

        changed_audit = dict(audit)
        changed_audit["summary"] = "Conflicting content for the same audit id."
        first_update = finish_slot(group, "first", 3)
        with self.assertRaises(EventConflictError):
            await self.store.compare_and_set_worker_group(
                first_update.model_dump(mode="json"),
                expected_revision=created.revision,
                audit_record=changed_audit,
            )

        persisted = await self.store.require_worker_group(group.group_id)
        self.assertEqual(persisted.revision, created.revision)
        persisted_group = StepWorkerGroup.model_validate(persisted.snapshot)
        persisted_attempt = next(
            slot.current_attempt for slot in persisted_group.slots
            if slot.assignment_key == "first"
        )
        self.assertEqual(persisted_attempt.status, WorkerAttemptStatus.RUNNING)
        audit_rows = await self.store.list_worker_attempt_audit(
            group_id=group.group_id
        )
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["summary"], "Original audit row.")

    async def test_concurrent_reporter_claim_has_one_new_owner(self):
        group = finish_slot(finish_slot(make_group(), "first", 3), "second", 4)
        await self.store.create_worker_group(group.model_dump(mode="json"))
        competing_store = AsyncEventStore(self.path)
        await competing_store.start()
        try:
            claims = await asyncio.gather(
                self.store.claim_worker_group_review(
                    group_id=group.group_id,
                    review_id="review-a",
                    changed_at=T0 + timedelta(seconds=5),
                ),
                competing_store.claim_worker_group_review(
                    group_id=group.group_id,
                    review_id="review-b",
                    changed_at=T0 + timedelta(seconds=5),
                ),
            )
            winners = [claim for claim in claims if claim.acquired_now]
            self.assertEqual(len(winners), 1)
            winner = winners[0]
            loser = claims[1] if claims[0] is winner else claims[0]
            self.assertTrue(winner.owned_by_caller)
            self.assertFalse(loser.owned_by_caller)
            self.assertEqual(
                winner.record.snapshot["review_status"],
                WorkerGroupReviewStatus.REVIEWING.value,
            )

            retry = await self.store.claim_worker_group_review(
                group_id=group.group_id,
                review_id=winner.record.review_id,
                changed_at=T0 + timedelta(seconds=6),
            )
            self.assertFalse(retry.acquired_now)
            self.assertTrue(retry.owned_by_caller)

            report = {"status": "COMPLETED", "summary": "Both sources found."}
            completed = await self.store.complete_worker_group_review(
                group_id=group.group_id,
                review_id=winner.record.review_id,
                report=report,
                changed_at=T0 + timedelta(seconds=7),
            )
            self.assertEqual(
                completed.snapshot["review_status"],
                WorkerGroupReviewStatus.REPORTED.value,
            )
            self.assertEqual(completed.report, report)

            replay = await self.store.complete_worker_group_review(
                group_id=group.group_id,
                review_id=winner.record.review_id,
                report=report,
                changed_at=T0 + timedelta(seconds=8),
            )
            self.assertEqual(replay, completed)
            with self.assertRaises(EventConflictError):
                await self.store.complete_worker_group_review(
                    group_id=group.group_id,
                    review_id="review-loser",
                    report=report,
                    changed_at=T0 + timedelta(seconds=9),
                )
        finally:
            await competing_store.close()

    async def test_reviewing_claim_survives_restart_without_reissue(self):
        group = finish_slot(finish_slot(make_group(), "first", 3), "second", 4)
        await self.store.create_worker_group(group.model_dump(mode="json"))
        first = await self.store.claim_worker_group_review(
            group_id=group.group_id,
            review_id="durable-owner",
            changed_at=T0 + timedelta(seconds=5),
        )
        self.assertTrue(first.acquired_now)

        await self.store.close()
        await self.store.start()
        competitor = await self.store.claim_worker_group_review(
            group_id=group.group_id,
            review_id="new-owner-after-restart",
            changed_at=T0 + timedelta(seconds=6),
        )
        self.assertFalse(competitor.acquired_now)
        self.assertFalse(competitor.owned_by_caller)
        self.assertEqual(
            competitor.record.snapshot["review_status"],
            WorkerGroupReviewStatus.REVIEWING.value,
        )

        owner_retry = await self.store.claim_worker_group_review(
            group_id=group.group_id,
            review_id="durable-owner",
            changed_at=T0 + timedelta(seconds=7),
        )
        self.assertFalse(owner_retry.acquired_now)
        self.assertTrue(owner_retry.owned_by_caller)


if __name__ == "__main__":
    unittest.main()
