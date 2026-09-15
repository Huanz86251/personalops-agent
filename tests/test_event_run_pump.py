"""Offline tests for persisted Event ingestion and serial execution."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from eventing import (
    AsyncEventStore,
    EventAction,
    EventOrigin,
    EventRunPump,
    EventStatus,
    RunStatus,
    create_agent_event,
    create_event_run,
)


class EventRunPumpTests(unittest.IsolatedAsyncioTestCase):
    async def test_insert_runs_before_older_pending_queue_event(self) -> None:
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            now = datetime.now(timezone.utc) - timedelta(seconds=2)
            queued = create_agent_event(
                event_id="evt_queue",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="ordinary task",
                origin=EventOrigin.FEISHU,
                received_at=now,
            )
            inserted = create_agent_event(
                event_id="evt_insert",
                conversation_id="conv_one",
                action=EventAction.INSERT,
                payload_text="priority task",
                origin=EventOrigin.FEISHU,
                received_at=now + timedelta(seconds=1),
            )
            await store.add_event(queued, run=create_event_run(queued))
            await store.add_event(inserted, run=create_event_run(inserted))

            handled: list[str] = []
            delivered: list[tuple[str, str]] = []
            failures: list[str] = []
            completed = asyncio.Event()

            async def handler(event, control, resume_from_checkpoint):
                self.assertFalse(resume_from_checkpoint)
                handled.append(event.event_id)
                return f"result:{event.payload_text}"

            async def on_result(event, result):
                delivered.append((event.event_id, result))
                if len(delivered) == 2:
                    completed.set()

            async def on_failure(event, error):
                failures.append(event.event_id)
                completed.set()

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
            )
            try:
                pump.start()
                await asyncio.wait_for(completed.wait(), timeout=2)
                self.assertEqual(handled, ["evt_insert", "evt_queue"])
                self.assertEqual(
                    [event_id for event_id, _ in delivered],
                    ["evt_insert", "evt_queue"],
                )
                self.assertEqual(failures, [])
                for event_id in ("evt_insert", "evt_queue"):
                    self.assertEqual(
                        (await store.require_event(event_id)).status,
                        EventStatus.APPLIED,
                    )
                    self.assertEqual(
                        (await store.require_run(event_id)).status,
                        RunStatus.COMPLETED,
                    )
            finally:
                await pump.stop()
                await store.close()

    async def test_handler_failure_closes_event_and_run(self) -> None:
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            event = create_agent_event(
                event_id="evt_failure",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="failing task",
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(event, run=create_event_run(event))
            failed = asyncio.Event()
            observed_error: list[str] = []

            async def handler(event, control, resume_from_checkpoint):
                self.assertFalse(resume_from_checkpoint)
                raise RuntimeError("planned failure")

            async def on_result(event, result):
                raise AssertionError("Failed events cannot deliver a result")

            async def on_failure(event, error):
                observed_error.append(str(error))
                failed.set()

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
            )
            try:
                pump.start()
                await asyncio.wait_for(failed.wait(), timeout=2)
                stored_event = await store.require_event(event.event_id)
                stored_run = await store.require_run(event.event_id)
                self.assertEqual(stored_event.status, EventStatus.FAILED)
                self.assertEqual(stored_event.result_code, "RUN_FAILED")
                self.assertEqual(stored_run.status, RunStatus.FAILED)
                self.assertEqual(observed_error, ["planned failure"])
            finally:
                await pump.stop()
                await store.close()

    async def test_insert_pauses_active_run_then_resumes_same_handler(self) -> None:
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            parent = create_agent_event(
                event_id="evt_parent",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="long task",
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(parent, run=create_event_run(parent))
            timeline: list[str] = []
            parent_started = asyncio.Event()
            completed = asyncio.Event()

            async def handler(event, control, resume_from_checkpoint):
                self.assertFalse(resume_from_checkpoint)
                if event.event_id == parent.event_id:
                    timeline.append("parent:start")
                    parent_started.set()
                    while not control.requested:
                        await asyncio.sleep(0)
                    await control.pause_point()
                    timeline.append("parent:resume")
                    return "parent-result"
                timeline.append("insert:run")
                return "insert-result"

            async def on_result(event, result):
                timeline.append(f"delivered:{event.event_id}")
                if event.event_id == parent.event_id:
                    completed.set()

            async def on_failure(event, error):
                self.fail(f"Unexpected failure: {event.event_id}: {error}")

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
            )
            try:
                pump.start()
                await asyncio.wait_for(parent_started.wait(), timeout=2)
                inserted = create_agent_event(
                    event_id="evt_insert_live",
                    conversation_id="conv_one",
                    action=EventAction.INSERT,
                    payload_text="urgent task",
                    target_event_id=parent.event_id,
                    origin=EventOrigin.FEISHU,
                )
                await store.add_event(inserted, run=create_event_run(inserted))
                pump.notify(inserted)
                await asyncio.wait_for(completed.wait(), timeout=2)

                self.assertEqual(
                    timeline,
                    [
                        "parent:start",
                        "insert:run",
                        "delivered:evt_insert_live",
                        "parent:resume",
                        "delivered:evt_parent",
                    ],
                )
                self.assertEqual(
                    (await store.require_run(parent.event_id)).status,
                    RunStatus.COMPLETED,
                )
                self.assertEqual(
                    (await store.require_run(inserted.event_id)).status,
                    RunStatus.COMPLETED,
                )
            finally:
                await pump.stop()
                await store.close()

    async def test_cancel_stops_active_run_at_safe_point_without_resuming(self) -> None:
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            target = create_agent_event(
                event_id="evt_cancelled_parent",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="long task",
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(target, run=create_event_run(target))
            target_started = asyncio.Event()
            cancel_delivered = asyncio.Event()
            timeline: list[str] = []

            async def handler(event, control, resume_from_checkpoint):
                self.assertEqual(event.event_id, target.event_id)
                timeline.append("target:start")
                target_started.set()
                while not control.requested:
                    await asyncio.sleep(0)
                timeline.append("target:safe-point")
                await control.pause_point()
                timeline.append("target:unexpected-resume")
                return "unexpected-result"

            async def on_result(event, result):
                timeline.append(f"delivered:{event.event_id}")
                if event.action is EventAction.CANCEL:
                    cancel_delivered.set()

            async def on_failure(event, error):
                self.fail(f"Unexpected failure: {event.event_id}: {error}")

            async def on_cancel(target_event, command_event):
                timeline.append(
                    f"archived:{target_event.event_id}:{command_event.event_id}"
                )

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
                on_cancel=on_cancel,
            )
            try:
                pump.start()
                await asyncio.wait_for(target_started.wait(), timeout=2)
                command = create_agent_event(
                    event_id="evt_cancel_control",
                    conversation_id="conv_one",
                    action=EventAction.CANCEL,
                    payload_text="",
                    target_event_id=target.event_id,
                    origin=EventOrigin.FEISHU,
                )
                await store.add_event(command)
                pump.notify(command)
                await asyncio.wait_for(cancel_delivered.wait(), timeout=2)

                self.assertEqual(
                    timeline,
                    [
                        "target:start",
                        "target:safe-point",
                        "archived:evt_cancelled_parent:evt_cancel_control",
                        "delivered:evt_cancel_control",
                    ],
                )
                self.assertEqual(
                    (await store.require_run(target.event_id)).status,
                    RunStatus.CANCELLED,
                )
                self.assertEqual(
                    (await store.require_event(target.event_id)).status,
                    EventStatus.APPLIED,
                )
                self.assertEqual(
                    (await store.require_event(command.event_id)).result_code,
                    "TARGET_CANCELLED",
                )
                self.assertIsNone(await store.get_run(command.event_id))
            finally:
                await pump.stop()
                await store.close()

    async def test_startup_resumes_insert_before_its_paused_parent(self) -> None:
        """A process crash must not lose either side of an INSERT rendezvous."""

        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            parent = create_agent_event(
                event_id="evt_parent_interrupted",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="long task",
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(parent, run=create_event_run(parent))
            claimed_parent = await store.claim_next_pending()
            self.assertEqual(claimed_parent.event_id, parent.event_id)
            await store.update_run_status(parent.event_id, RunStatus.RUNNING)
            await store.update_run_status(parent.event_id, RunStatus.PAUSED)

            inserted = create_agent_event(
                event_id="evt_insert_interrupted",
                conversation_id="conv_one",
                action=EventAction.INSERT,
                payload_text="urgent task",
                target_event_id=parent.event_id,
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(inserted, run=create_event_run(inserted))
            claimed_insert = await store.claim_pending_insert(
                target_event_id=parent.event_id
            )
            self.assertEqual(claimed_insert.event_id, inserted.event_id)
            await store.update_run_status(inserted.event_id, RunStatus.RUNNING)

            handled: list[tuple[str, bool]] = []
            delivered: list[str] = []
            completed = asyncio.Event()

            async def handler(event, control, resume_from_checkpoint):
                handled.append((event.event_id, resume_from_checkpoint))
                return f"result:{event.event_id}"

            async def on_result(event, result):
                delivered.append(event.event_id)
                if len(delivered) == 2:
                    completed.set()

            async def on_failure(event, error):
                self.fail(f"Unexpected failure: {event.event_id}: {error}")

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
            )
            try:
                pump.start()
                await asyncio.wait_for(completed.wait(), timeout=2)
                self.assertEqual(
                    handled,
                    [
                        (inserted.event_id, True),
                        (parent.event_id, True),
                    ],
                )
                self.assertEqual(
                    delivered,
                    [inserted.event_id, parent.event_id],
                )
                for event_id in (inserted.event_id, parent.event_id):
                    self.assertEqual(
                        (await store.require_event(event_id)).status,
                        EventStatus.APPLIED,
                    )
                    self.assertEqual(
                        (await store.require_run(event_id)).status,
                        RunStatus.COMPLETED,
                    )
            finally:
                await pump.stop()
                await store.close()

    async def test_startup_applies_cancel_before_resuming_orphaned_target(self) -> None:
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            target = create_agent_event(
                event_id="evt_orphaned_cancel_target",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="interrupted task",
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(target, run=create_event_run(target))
            await store.claim_next_pending()
            await store.update_run_status(target.event_id, RunStatus.RUNNING)
            command = create_agent_event(
                event_id="evt_startup_cancel",
                conversation_id="conv_one",
                action=EventAction.CANCEL,
                payload_text="",
                target_event_id=target.event_id,
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(command)
            delivered = asyncio.Event()
            handler_calls: list[str] = []

            async def handler(event, control, resume_from_checkpoint):
                handler_calls.append(event.event_id)
                return "must-not-run"

            async def on_result(event, result):
                if event.event_id == command.event_id:
                    delivered.set()

            async def on_failure(event, error):
                self.fail(f"Unexpected failure: {event.event_id}: {error}")

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
            )
            try:
                pump.start()
                await asyncio.wait_for(delivered.wait(), timeout=2)
                self.assertEqual(handler_calls, [])
                self.assertEqual(
                    (await store.require_run(target.event_id)).status,
                    RunStatus.CANCELLED,
                )
                self.assertEqual(
                    (await store.require_event(command.event_id)).result_code,
                    "TARGET_CANCELLED",
                )
            finally:
                await pump.stop()
                await store.close()

    async def test_replace_stops_active_generation_and_runs_replacement(self) -> None:
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            target = create_agent_event(
                event_id="evt_replace_parent",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="old task",
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(target, run=create_event_run(target))
            target_started = asyncio.Event()
            replacement_delivered = asyncio.Event()
            timeline: list[str] = []

            async def handler(event, control, resume_from_checkpoint):
                self.assertFalse(resume_from_checkpoint)
                if event.event_id == target.event_id:
                    timeline.append("target:start")
                    target_started.set()
                    while not control.requested:
                        await asyncio.sleep(0)
                    timeline.append("target:safe-point")
                    await control.pause_point()
                    timeline.append("target:unexpected-resume")
                    return "unexpected-old-result"
                timeline.append("replacement:start")
                return "replacement-result"

            async def on_result(event, result):
                timeline.append(f"delivered:{event.event_id}:{result}")
                if event.action is EventAction.REPLACE:
                    replacement_delivered.set()

            async def on_failure(event, error):
                self.fail(f"Unexpected failure: {event.event_id}: {error}")

            async def on_replace(target_event, replacement_event):
                timeline.append(
                    f"archived:{target_event.event_id}:{replacement_event.event_id}"
                )

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
                on_replace=on_replace,
            )
            try:
                pump.start()
                await asyncio.wait_for(target_started.wait(), timeout=2)
                replacement = create_agent_event(
                    event_id="evt_replace_command",
                    conversation_id="conv_one",
                    action=EventAction.REPLACE,
                    payload_text="new task",
                    target_event_id=target.event_id,
                    origin=EventOrigin.FEISHU,
                )
                await store.add_event(
                    replacement,
                    run=create_event_run(replacement),
                )
                pump.notify(replacement)
                await asyncio.wait_for(replacement_delivered.wait(), timeout=2)

                self.assertEqual(
                    timeline,
                    [
                        "target:start",
                        "target:safe-point",
                        "archived:evt_replace_parent:evt_replace_command",
                        "replacement:start",
                        "delivered:evt_replace_command:replacement-result",
                    ],
                )
                self.assertEqual(
                    (await store.require_run(target.event_id)).status,
                    RunStatus.SUPERSEDED,
                )
                self.assertEqual(
                    (await store.require_run(replacement.event_id)).status,
                    RunStatus.COMPLETED,
                )
                self.assertEqual(
                    (await store.require_event(replacement.event_id)).status,
                    EventStatus.APPLIED,
                )
            finally:
                await pump.stop()
                await store.close()

    async def test_startup_replace_wins_before_orphaned_target_resumes(self) -> None:
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            target = create_agent_event(
                event_id="evt_orphaned_replace_target",
                conversation_id="conv_one",
                action=EventAction.QUEUE,
                payload_text="interrupted old task",
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(target, run=create_event_run(target))
            await store.claim_next_pending()
            await store.update_run_status(target.event_id, RunStatus.RUNNING)
            replacement = create_agent_event(
                event_id="evt_startup_replace",
                conversation_id="conv_one",
                action=EventAction.REPLACE,
                payload_text="new task after restart",
                target_event_id=target.event_id,
                origin=EventOrigin.FEISHU,
            )
            await store.add_event(
                replacement,
                run=create_event_run(replacement),
            )
            delivered = asyncio.Event()
            handler_calls: list[tuple[str, bool]] = []

            async def handler(event, control, resume_from_checkpoint):
                handler_calls.append((event.event_id, resume_from_checkpoint))
                return "replacement-result"

            async def on_result(event, result):
                if event.event_id == replacement.event_id:
                    delivered.set()

            async def on_failure(event, error):
                self.fail(f"Unexpected failure: {event.event_id}: {error}")

            pump = EventRunPump(
                store,
                handler=handler,
                on_result=on_result,
                on_failure=on_failure,
            )
            try:
                pump.start()
                await asyncio.wait_for(delivered.wait(), timeout=2)
                self.assertEqual(handler_calls, [(replacement.event_id, False)])
                self.assertEqual(
                    (await store.require_run(target.event_id)).status,
                    RunStatus.SUPERSEDED,
                )
                self.assertEqual(
                    (await store.require_run(replacement.event_id)).status,
                    RunStatus.COMPLETED,
                )
            finally:
                await pump.stop()
                await store.close()


if __name__ == "__main__":
    unittest.main()
