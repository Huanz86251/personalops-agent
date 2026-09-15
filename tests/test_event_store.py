"""Offline SQLite tests for the durable Event Store."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from eventing import (
    AsyncEventStore,
    EventAction,
    EventOrigin,
    EventStatus,
    RunStatus,
    create_agent_event,
    create_event_run,
)


class EventStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.store_path = Path(self.temporary.name) / "events.sqlite3"
        self.store = AsyncEventStore(self.store_path)
        await self.store.start()
        self.now = datetime(2026, 9, 2, 2, 0, tzinfo=timezone.utc)

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.temporary.cleanup()

    def make_event(
        self,
        event_id: str,
        *,
        action: EventAction = EventAction.QUEUE,
        received_offset: int = 0,
        target_event_id: str | None = None,
    ):
        payload = "" if action is EventAction.CANCEL else f"任务 {event_id}"
        return create_agent_event(
            event_id=event_id,
            conversation_id="conversation_1",
            action=action,
            payload_text=payload,
            target_event_id=target_event_id,
            origin=EventOrigin.FEISHU,
            received_at=self.now + timedelta(seconds=received_offset),
        )

    async def test_event_and_run_are_inserted_and_reopened(self) -> None:
        event = self.make_event("evt_a")
        run = create_event_run(event)
        await self.store.add_event(event, run=run)

        await self.store.close()
        await self.store.start()

        self.assertEqual(await self.store.get_event("evt_a"), event)
        self.assertEqual(await self.store.get_run("evt_a"), run)

    async def test_oldest_pending_event_is_claimed_atomically(self) -> None:
        later = self.make_event("evt_later", received_offset=2)
        earlier = self.make_event("evt_earlier", received_offset=1)
        await self.store.add_event(later)
        await self.store.add_event(earlier)

        claimed = await self.store.claim_next_pending(
            changed_at=self.now + timedelta(seconds=3)
        )

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.event_id, "evt_earlier")
        self.assertEqual(claimed.status, EventStatus.HANDLING)
        pending = await self.store.list_events(statuses=[EventStatus.PENDING])
        self.assertEqual([item.event_id for item in pending], ["evt_later"])

    async def test_applied_means_command_applied_not_run_completed(self) -> None:
        event = self.make_event("evt_applied")
        run = create_event_run(event)
        await self.store.add_event(event, run=run)
        await self.store.update_event_status(
            event.event_id,
            EventStatus.HANDLING,
            changed_at=self.now + timedelta(seconds=1),
        )
        applied = await self.store.update_event_status(
            event.event_id,
            EventStatus.APPLIED,
            changed_at=self.now + timedelta(seconds=2),
            result_code="RUN_CREATED",
        )

        persisted_run = await self.store.require_run(event.event_id)
        self.assertEqual(applied.status, EventStatus.APPLIED)
        self.assertEqual(persisted_run.status, RunStatus.QUEUED)

    async def test_run_pause_and_resume_are_persisted_separately(self) -> None:
        event = self.make_event("evt_run")
        await self.store.add_event(event, run=create_event_run(event))
        await self.store.update_run_status(
            event.event_id,
            RunStatus.RUNNING,
            changed_at=self.now + timedelta(seconds=1),
        )
        paused = await self.store.update_run_status(
            event.event_id,
            RunStatus.PAUSED,
            changed_at=self.now + timedelta(seconds=2),
        )
        resumed = await self.store.update_run_status(
            event.event_id,
            RunStatus.RUNNING,
            changed_at=self.now + timedelta(seconds=3),
        )

        self.assertEqual(paused.status, RunStatus.PAUSED)
        self.assertEqual(resumed.status, RunStatus.RUNNING)
        self.assertEqual(await self.store.get_run(event.event_id), resumed)

    async def test_cancel_closes_target_and_command_in_one_transaction(self) -> None:
        target = self.make_event("evt_cancel_target")
        await self.store.add_event(target, run=create_event_run(target))
        await self.store.update_event_status(
            target.event_id,
            EventStatus.HANDLING,
            changed_at=self.now + timedelta(seconds=1),
        )
        await self.store.update_run_status(
            target.event_id,
            RunStatus.RUNNING,
            changed_at=self.now + timedelta(seconds=1),
        )
        command = self.make_event(
            "evt_cancel_command",
            action=EventAction.CANCEL,
            target_event_id=target.event_id,
            received_offset=2,
        )
        await self.store.add_event(command)

        applied = await self.store.apply_cancel(
            command.event_id,
            changed_at=self.now + timedelta(seconds=3),
        )

        self.assertTrue(applied.target_was_cancelled)
        self.assertEqual(applied.target_run.status, RunStatus.CANCELLED)
        self.assertEqual(applied.target_event.status, EventStatus.APPLIED)
        self.assertEqual(
            applied.target_event.result_code,
            "RUN_CANCELLED_BY_USER",
        )
        self.assertEqual(applied.command_event.status, EventStatus.APPLIED)
        self.assertEqual(
            applied.command_event.result_code,
            "TARGET_CANCELLED",
        )
        self.assertIsNone(await self.store.get_run(command.event_id))

    async def test_cancel_of_completed_target_is_consumed_as_noop(self) -> None:
        target = self.make_event("evt_finished_target")
        await self.store.add_event(target, run=create_event_run(target))
        await self.store.update_event_status(
            target.event_id,
            EventStatus.HANDLING,
            changed_at=self.now + timedelta(seconds=1),
        )
        await self.store.update_run_status(
            target.event_id,
            RunStatus.RUNNING,
            changed_at=self.now + timedelta(seconds=1),
        )
        await self.store.update_run_status(
            target.event_id,
            RunStatus.COMPLETED,
            changed_at=self.now + timedelta(seconds=2),
        )
        await self.store.update_event_status(
            target.event_id,
            EventStatus.APPLIED,
            changed_at=self.now + timedelta(seconds=2),
            result_code="RUN_COMPLETED",
        )
        command = self.make_event(
            "evt_late_cancel",
            action=EventAction.CANCEL,
            target_event_id=target.event_id,
            received_offset=3,
        )
        await self.store.add_event(command)

        applied = await self.store.apply_cancel(
            command.event_id,
            changed_at=self.now + timedelta(seconds=4),
        )

        self.assertFalse(applied.target_was_cancelled)
        self.assertEqual(applied.target_run.status, RunStatus.COMPLETED)
        self.assertEqual(
            applied.command_event.result_code,
            "TARGET_ALREADY_COMPLETED",
        )

    async def test_only_one_pending_cancel_may_target_a_run(self) -> None:
        target = self.make_event("evt_single_cancel_target")
        await self.store.add_event(target, run=create_event_run(target))
        first = self.make_event(
            "evt_first_cancel",
            action=EventAction.CANCEL,
            target_event_id=target.event_id,
            received_offset=1,
        )
        second = self.make_event(
            "evt_second_cancel",
            action=EventAction.CANCEL,
            target_event_id=target.event_id,
            received_offset=2,
        )
        await self.store.add_event(first)

        with self.assertRaisesRegex(Exception, "pending terminal control"):
            await self.store.add_event(second)

        self.assertIsNone(await self.store.get_event(second.event_id))

    async def test_replace_supersedes_target_and_keeps_new_run_queued(self) -> None:
        target = self.make_event("evt_replace_target")
        await self.store.add_event(target, run=create_event_run(target))
        await self.store.update_event_status(
            target.event_id,
            EventStatus.HANDLING,
            changed_at=self.now + timedelta(seconds=1),
        )
        await self.store.update_run_status(
            target.event_id,
            RunStatus.RUNNING,
            changed_at=self.now + timedelta(seconds=1),
        )
        replacement = self.make_event(
            "evt_replacement",
            action=EventAction.REPLACE,
            target_event_id=target.event_id,
            received_offset=2,
        )
        await self.store.add_event(
            replacement,
            run=create_event_run(replacement),
        )

        applied = await self.store.apply_replace(
            replacement.event_id,
            changed_at=self.now + timedelta(seconds=3),
        )

        self.assertTrue(applied.target_was_superseded)
        self.assertEqual(applied.target_run.status, RunStatus.SUPERSEDED)
        self.assertEqual(applied.target_event.status, EventStatus.APPLIED)
        self.assertEqual(
            applied.target_event.result_code,
            "RUN_SUPERSEDED_BY_USER",
        )
        self.assertEqual(applied.replacement_event.status, EventStatus.HANDLING)
        self.assertEqual(
            applied.replacement_event.result_code,
            "TARGET_SUPERSEDED",
        )
        self.assertEqual(applied.replacement_run.status, RunStatus.QUEUED)

    async def test_target_accepts_only_one_pending_cancel_or_replace(self) -> None:
        target = self.make_event("evt_control_target")
        await self.store.add_event(target, run=create_event_run(target))
        replacement = self.make_event(
            "evt_pending_replace",
            action=EventAction.REPLACE,
            target_event_id=target.event_id,
            received_offset=1,
        )
        cancel = self.make_event(
            "evt_conflicting_cancel",
            action=EventAction.CANCEL,
            target_event_id=target.event_id,
            received_offset=2,
        )
        await self.store.add_event(
            replacement,
            run=create_event_run(replacement),
        )

        with self.assertRaisesRegex(Exception, "pending terminal control"):
            await self.store.add_event(cancel)

        self.assertIsNone(await self.store.get_event(cancel.event_id))

    async def test_insert_target_is_enforced_by_foreign_key(self) -> None:
        insert = self.make_event(
            "evt_insert",
            action=EventAction.INSERT,
            target_event_id="evt_missing",
        )

        with self.assertRaisesRegex(Exception, "FOREIGN KEY constraint failed"):
            await self.store.add_event(insert)

        self.assertIsNone(await self.store.get_event("evt_insert"))

    async def test_handling_event_can_be_requeued_for_recovery(self) -> None:
        event = self.make_event("evt_recovery")
        await self.store.add_event(event)
        await self.store.update_event_status(
            event.event_id,
            EventStatus.HANDLING,
            changed_at=self.now + timedelta(seconds=1),
        )
        recovered = await self.store.update_event_status(
            event.event_id,
            EventStatus.PENDING,
            changed_at=self.now + timedelta(seconds=2),
            result_code="RECOVERED_AFTER_RESTART",
        )

        self.assertEqual(recovered.status, EventStatus.PENDING)
        self.assertEqual(recovered.result_code, "RECOVERED_AFTER_RESTART")

    async def test_worker_progress_inbox_is_append_only_and_cursor_is_explicit(
        self,
    ) -> None:
        record = {
            "sequence": 1,
            "total_tool_calls": 5,
            "published_at": "2026-09-02T02:00:00Z",
            "worker_id": "worker-1",
            "event_id": "event-1",
            "step_id": "1",
            "progress": {
                "phase": "searching",
                "summary": "Found one primary source.",
                "findings": ["Source is reachable."],
                "difficulties": [],
                "evidence_refs": ["https://example.com"],
                "next_action": "Verify the claim.",
                "completion_claim": "not_ready",
            },
        }

        self.assertTrue(await self.store.append_worker_progress(record))
        self.assertFalse(await self.store.append_worker_progress(record))
        self.assertEqual(
            await self.store.list_worker_progress(worker_id="worker-1"),
            [record],
        )
        self.assertEqual(
            await self.store.get_worker_progress_cursor(
                consumer_id="planning",
                worker_id="worker-1",
            ),
            0,
        )
        self.assertEqual(
            await self.store.advance_worker_progress_cursor(
                consumer_id="planning",
                worker_id="worker-1",
                sequence=1,
                changed_at=self.now,
            ),
            1,
        )

        changed = dict(record)
        changed["total_tool_calls"] = 6
        with self.assertRaisesRegex(Exception, "different content"):
            await self.store.append_worker_progress(changed)

    async def test_leadership_wake_directive_and_cursor_commit_atomically(self):
        wake = {
            "wake_id": "wake-1",
            "event_id": "event-1",
            "reason": "SINGLE_WORKER_THRESHOLD",
            "created_at": "2026-09-02T02:00:00Z",
            "workers": [],
        }
        directive = {
            "decision": {
                "action": "GUIDE",
                "reason": "Adjust the query.",
                "target_worker_ids": ["worker-1"],
                "guidance": "Use the official source.",
                "replacement_assignment": None,
            },
            "model_rounds_used": 1,
            "used_fallback": False,
            "wake_id": "wake-1",
        }

        await self.store.save_leadership_wake(
            wake=wake,
            result=directive,
            directives={"worker-1": directive},
            cursor_updates={"worker-1": 2},
            consumer_id="planning_supervisor",
        )

        self.assertEqual(
            await self.store.get_worker_progress_cursor(
                consumer_id="planning_supervisor",
                worker_id="worker-1",
            ),
            2,
        )
        pending = await self.store.get_pending_worker_directive("worker-1")
        self.assertEqual(pending["wake_id"], "wake-1")
        self.assertEqual(
            pending["directive"]["decision"]["action"],
            "GUIDE",
        )

        await self.store.mark_worker_directive_applied(
            wake_id="wake-1",
            worker_id="worker-1",
            changed_at=self.now,
        )
        self.assertIsNone(
            await self.store.get_pending_worker_directive("worker-1")
        )


if __name__ == "__main__":
    unittest.main()
