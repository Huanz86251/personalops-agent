"""Offline tests for bounded, append-only Worker attempt audit records."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from eventing import AsyncEventStore, EventConflictError
from workers.audit import WorkerAttemptAuditRecord


class WorkerAttemptAuditTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.store = AsyncEventStore(
            Path(self.temporary.name) / "events.sqlite3"
        )
        await self.store.start()
        self.now = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.temporary.cleanup()

    def stale_callback(self) -> WorkerAttemptAuditRecord:
        return WorkerAttemptAuditRecord(
            audit_id="callback-old-attempt-1",
            group_id="group-audit",
            assignment_key="official",
            attempt_no=1,
            worker_id="worker:group-audit:slot:official:attempt:1",
            workspace_id="workspace:group-audit:slot:official:attempt:1",
            checkpoint_thread_id=(
                "event-audit:group-audit:slot:official:attempt:1"
            ),
            operation="CALLBACK",
            outcome="STALE",
            occurred_at=self.now,
            recorded_at=self.now + timedelta(seconds=2),
            summary=(
                "Late completion from replaced attempt; retained for audit "
                "and excluded from the current slot."
            ),
            trace_id="trace-old-attempt",
            status_before="REPLACED",
            status_after="REPLACED",
        )

    async def test_stale_callback_index_is_persisted_without_raw_result(self):
        record = self.stale_callback()
        payload = record.model_dump(mode="json")
        self.assertTrue(await self.store.append_worker_attempt_audit(payload))
        self.assertFalse(await self.store.append_worker_attempt_audit(payload))

        rows = await self.store.list_worker_attempt_audit(
            group_id="group-audit"
        )
        self.assertEqual(rows, [payload])
        self.assertNotIn("messages", rows[0])
        self.assertNotIn("tool_result", rows[0])
        self.assertEqual(rows[0]["outcome"], "STALE")

    async def test_reusing_audit_id_for_different_content_is_a_conflict(self):
        payload = self.stale_callback().model_dump(mode="json")
        await self.store.append_worker_attempt_audit(payload)
        changed = dict(payload)
        changed["summary"] = "Different callback content."
        with self.assertRaises(EventConflictError):
            await self.store.append_worker_attempt_audit(changed)

    async def test_assignment_filter_does_not_mix_other_slot_events(self):
        first = self.stale_callback().model_dump(mode="json")
        second = dict(first)
        second.update(
            audit_id="callback-other-slot",
            assignment_key="reviews",
            worker_id="worker:group-audit:slot:reviews:attempt:1",
            workspace_id="workspace:group-audit:slot:reviews:attempt:1",
            checkpoint_thread_id=(
                "event-audit:group-audit:slot:reviews:attempt:1"
            ),
        )
        await self.store.append_worker_attempt_audit(first)
        await self.store.append_worker_attempt_audit(second)

        rows = await self.store.list_worker_attempt_audit(
            group_id="group-audit",
            assignment_key="official",
        )
        self.assertEqual(rows, [first])


if __name__ == "__main__":
    unittest.main()
