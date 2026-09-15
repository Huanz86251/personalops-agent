"""Provider-free tests for per-slot Worker attempt generations."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from planning_models import CodeRequirement, CodeTaskContract, PlanStep, WorkerAssignment
from workers.group import (
    StepWorkerGroup,
    WorkerAttemptStatus,
    WorkerGroupReviewStatus,
    WorkerWorkspaceStatus,
    create_step_worker_group,
    finish_worker_attempt,
    replace_worker_attempt,
    start_worker_attempt,
)


T0 = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)


def parallel_step() -> PlanStep:
    return PlanStep(
        step_id=2,
        objective="Cross-check cinema information.",
        success_criteria=["Return complementary cited findings."],
        worker_kind="WEB",
        execution_mode="PARALLEL",
        worker_assignments=[
            WorkerAssignment(
                assignment_key="official",
                objective="Check official listings.",
            ),
            WorkerAssignment(
                assignment_key="reviews",
                objective="Check independent reviews.",
            ),
            WorkerAssignment(
                assignment_key="prices",
                objective="Cross-check current prices.",
            ),
        ],
    )


def current(group: StepWorkerGroup, key: str):
    return next(
        slot.current_attempt
        for slot in group.slots
        if slot.assignment_key == key
    )


def start(group: StepWorkerGroup, key: str, seconds: int) -> StepWorkerGroup:
    attempt = current(group, key)
    return start_worker_attempt(
        group,
        assignment_key=key,
        attempt_no=attempt.attempt_no,
        worker_id=attempt.worker_id,
        changed_at=T0 + timedelta(seconds=seconds),
    ).group


def finish(
    group: StepWorkerGroup,
    key: str,
    status: WorkerAttemptStatus,
    seconds: int,
) -> StepWorkerGroup:
    attempt = current(group, key)
    return finish_worker_attempt(
        group,
        assignment_key=key,
        attempt_no=attempt.attempt_no,
        worker_id=attempt.worker_id,
        status=status,
        terminal_reason=f"{key} stopped as {status.value}",
        changed_at=T0 + timedelta(seconds=seconds),
    ).group


class WorkerGroupTests(unittest.TestCase):
    def test_out_of_order_completion_opens_barrier_only_after_last_slot(self):
        group = create_step_worker_group(
            parallel_step(),
            event_id="event-1",
            group_id="group-1",
            created_at=T0,
        )
        self.assertEqual(group.active_worker_ids, ())
        self.assertEqual(len(group.nonterminal_worker_ids), 3)
        self.assertEqual(
            len({attempt.workspace.workspace_id for attempt in group.current_attempts}),
            3,
        )
        self.assertEqual(
            len(
                {
                    attempt.workspace.checkpoint_thread_id
                    for attempt in group.current_attempts
                }
            ),
            3,
        )
        self.assertTrue(
            all(
                attempt.workspace.status is WorkerWorkspaceStatus.PREPARED
                for attempt in group.current_attempts
            )
        )
        for offset, key in enumerate(("official", "reviews", "prices"), 1):
            group = start(group, key, offset)

        self.assertTrue(
            all(
                attempt.workspace.status is WorkerWorkspaceStatus.ACTIVE
                for attempt in group.current_attempts
            )
        )

        group = finish(group, "prices", WorkerAttemptStatus.SUBMITTED, 4)
        group = finish(group, "official", WorkerAttemptStatus.SUBMITTED, 5)
        self.assertEqual(group.review_status, WorkerGroupReviewStatus.WAITING)
        self.assertEqual(len(group.active_worker_ids), 1)

        group = finish(group, "reviews", WorkerAttemptStatus.NATURAL_EXIT, 6)
        self.assertEqual(group.review_status, WorkerGroupReviewStatus.JOIN_READY)
        self.assertEqual(group.active_worker_ids, ())

    def test_failed_cancelled_and_budget_exhausted_are_terminal_not_success(self):
        group = create_step_worker_group(
            parallel_step(),
            event_id="event-2",
            group_id="group-2",
            created_at=T0,
        )
        for offset, key in enumerate(("official", "reviews", "prices"), 1):
            group = start(group, key, offset)
        group = finish(group, "official", WorkerAttemptStatus.FAILED, 4)
        group = finish(group, "reviews", WorkerAttemptStatus.CANCELLED, 5)
        group = finish(group, "prices", WorkerAttemptStatus.BUDGET_EXHAUSTED, 6)

        self.assertEqual(group.review_status, WorkerGroupReviewStatus.JOIN_READY)
        self.assertEqual(
            [attempt.status for attempt in group.current_attempts],
            [
                WorkerAttemptStatus.FAILED,
                WorkerAttemptStatus.CANCELLED,
                WorkerAttemptStatus.BUDGET_EXHAUSTED,
            ],
        )
        self.assertTrue(
            all(
                attempt.workspace.status is WorkerWorkspaceStatus.FROZEN
                and attempt.workspace.retain_until is not None
                for attempt in group.current_attempts
            )
        )

    def test_replace_appends_generation_and_late_old_callback_is_stale(self):
        group = create_step_worker_group(
            parallel_step(),
            event_id="event-3",
            group_id="group-3",
            created_at=T0,
        )
        group = start(group, "official", 1)
        old_attempt = current(group, "official")

        mutation = replace_worker_attempt(
            group,
            assignment_key="official",
            attempt_no=old_attempt.attempt_no,
            worker_id=old_attempt.worker_id,
            replacement_objective="Use a different official query.",
            reason="The original query was off target.",
            changed_at=T0 + timedelta(seconds=2),
        )
        group = mutation.group
        slot = next(slot for slot in group.slots if slot.assignment_key == "official")
        self.assertTrue(mutation.applied)
        self.assertEqual(len(slot.attempts), 2)
        self.assertEqual(slot.attempts[0].status, WorkerAttemptStatus.REPLACED)
        self.assertEqual(
            slot.attempts[0].workspace.status,
            WorkerWorkspaceStatus.FROZEN,
        )
        self.assertEqual(
            slot.attempts[0].workspace.retain_until,
            T0 + timedelta(days=1, seconds=2),
        )
        self.assertEqual(slot.current_attempt.attempt_no, 2)
        self.assertEqual(slot.current_attempt.status, WorkerAttemptStatus.PENDING)
        self.assertEqual(
            slot.current_attempt.workspace.status,
            WorkerWorkspaceStatus.PREPARED,
        )
        self.assertNotEqual(
            slot.attempts[0].workspace.workspace_id,
            slot.current_attempt.workspace.workspace_id,
        )
        self.assertNotEqual(
            slot.attempts[0].workspace.checkpoint_thread_id,
            slot.current_attempt.workspace.checkpoint_thread_id,
        )
        self.assertFalse(
            slot.attempts[0].workspace.cleanup_due(T0 + timedelta(hours=23))
        )
        self.assertTrue(
            slot.attempts[0].workspace.cleanup_due(
                T0 + timedelta(days=1, seconds=2)
            )
        )

        late = finish_worker_attempt(
            group,
            assignment_key="official",
            attempt_no=old_attempt.attempt_no,
            worker_id=old_attempt.worker_id,
            status=WorkerAttemptStatus.SUBMITTED,
            terminal_reason="Late obsolete result.",
            changed_at=T0 + timedelta(seconds=3),
        )
        self.assertFalse(late.applied)
        self.assertTrue(late.stale)
        self.assertEqual(late.group, group)

    def test_latest_generation_controls_join_barrier_after_replace(self):
        group = create_step_worker_group(
            parallel_step(),
            event_id="event-4",
            group_id="group-4",
            created_at=T0,
        )
        group = start(group, "official", 1)
        old_attempt = current(group, "official")
        group = replace_worker_attempt(
            group,
            assignment_key="official",
            attempt_no=old_attempt.attempt_no,
            worker_id=old_attempt.worker_id,
            replacement_objective="Retry official listings with a new query.",
            reason="Leadership REPLACE.",
            changed_at=T0 + timedelta(seconds=2),
        ).group
        group = start(group, "official", 3)
        group = start(group, "reviews", 4)
        group = start(group, "prices", 5)
        group = finish(group, "reviews", WorkerAttemptStatus.SUBMITTED, 6)
        group = finish(group, "prices", WorkerAttemptStatus.SUBMITTED, 7)
        self.assertEqual(group.review_status, WorkerGroupReviewStatus.WAITING)
        group = finish(group, "official", WorkerAttemptStatus.SUBMITTED, 8)
        self.assertEqual(group.review_status, WorkerGroupReviewStatus.JOIN_READY)

    def test_unknown_callback_identity_is_rejected_not_marked_stale(self):
        group = create_step_worker_group(
            parallel_step(),
            event_id="event-unknown",
            group_id="group-unknown",
            created_at=T0,
        )
        with self.assertRaisesRegex(ValueError, "does not belong"):
            finish_worker_attempt(
                group,
                assignment_key="official",
                attempt_no=99,
                worker_id="made-up-worker",
                status=WorkerAttemptStatus.FAILED,
                terminal_reason="Not a real callback.",
                changed_at=T0 + timedelta(seconds=1),
            )

    def test_late_delivery_with_earlier_event_time_is_still_applied(self):
        group = create_step_worker_group(
            parallel_step(),
            event_id="event-reordered",
            group_id="group-reordered",
            created_at=T0,
        )
        for offset, key in enumerate(("official", "reviews", "prices"), 1):
            group = start(group, key, offset)
        group = finish(group, "prices", WorkerAttemptStatus.SUBMITTED, 8)
        group = finish(group, "official", WorkerAttemptStatus.SUBMITTED, 6)

        self.assertEqual(
            current(group, "official").status_changed_at,
            T0 + timedelta(seconds=6),
        )
        self.assertEqual(group.updated_at, T0 + timedelta(seconds=8))

    def test_single_step_uses_the_same_group_contract(self):
        step = PlanStep(
            step_id=1,
            objective="Modify one local file.",
            success_criteria=["Focused tests pass."],
            worker_kind="CODE",
            code_task=CodeTaskContract(
                requirements=[
                    CodeRequirement(
                        requirement_id="focused_tests",
                        statement="Focused tests pass.",
                    )
                ]
            ),
        )
        group = create_step_worker_group(
            step,
            event_id="event-single",
            group_id="group-single",
            created_at=T0,
        )
        self.assertEqual(len(group.slots), 1)
        self.assertEqual(group.slots[0].assignment_key, "primary")

    def test_group_snapshot_round_trips_as_json(self):
        group = create_step_worker_group(
            parallel_step(),
            event_id="event-json",
            group_id="group-json",
            created_at=T0,
        )
        restored = StepWorkerGroup.model_validate_json(group.model_dump_json())
        self.assertEqual(restored, group)


if __name__ == "__main__":
    unittest.main()
