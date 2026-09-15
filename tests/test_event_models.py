"""Offline tests for the durable event-domain model."""

import json
from datetime import datetime, timedelta, timezone
import unittest

from eventing import (
    AgentEvent,
    EventAction,
    EventOrigin,
    EventRun,
    EventStatus,
    RunStatus,
    build_planning_thread_id,
    create_agent_event,
    create_event_run,
    transition_event_status,
    transition_run_status,
)


class EventModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.received_at = datetime(2026, 9, 1, 2, 0, tzinfo=timezone.utc)

    def test_queue_event_uses_one_id_for_event_and_run(self) -> None:
        event = create_agent_event(
            event_id="evt_queue",
            conversation_id="conversation_1",
            action=EventAction.QUEUE,
            payload_text="整理今天的文件",
            origin=EventOrigin.FEISHU,
            received_at=self.received_at,
        )
        run = create_event_run(event)

        self.assertEqual(event.status, EventStatus.PENDING)
        self.assertEqual(event.received_at, event.status_changed_at)
        self.assertEqual(run.event_id, event.event_id)
        self.assertEqual(run.status, RunStatus.QUEUED)
        self.assertEqual(
            build_planning_thread_id(event),
            "planning:conversation_1:evt_queue",
        )

    def test_insert_targets_the_affected_event_not_the_previous_message(self) -> None:
        event = create_agent_event(
            event_id="evt_insert",
            conversation_id="conversation_1",
            action=EventAction.INSERT,
            payload_text="查询服务器状态",
            target_event_id="evt_running",
            origin=EventOrigin.DESKTOP,
            received_at=self.received_at,
        )

        self.assertEqual(event.target_event_id, "evt_running")
        self.assertEqual(event.origin, EventOrigin.DESKTOP)

    def test_event_dict_round_trip_is_json_and_checkpoint_safe(self) -> None:
        event = create_agent_event(
            event_id="evt_round_trip",
            conversation_id="conversation_1",
            action=EventAction.REPLACE,
            payload_text="处理服务器报警",
            target_event_id="evt_old",
            origin=EventOrigin.FEISHU,
            reply_target_id="oc_reply_target",
            received_at=self.received_at,
        )

        encoded = json.dumps(event.to_dict(), ensure_ascii=False)
        restored = AgentEvent.from_dict(json.loads(encoded))

        self.assertEqual(restored, event)
        self.assertEqual(restored.reply_target_id, "oc_reply_target")
        self.assertEqual(event.to_dict()["received_at"], "2026-09-01T02:00:00Z")

    def test_replace_requires_an_explicit_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "REPLACE events require"):
            create_agent_event(
                event_id="evt_replace_without_target",
                conversation_id="conversation_1",
                action=EventAction.REPLACE,
                payload_text="改成新的任务",
                origin=EventOrigin.FEISHU,
                received_at=self.received_at,
            )

    def test_event_status_time_moves_with_legal_transitions(self) -> None:
        event = create_agent_event(
            event_id="evt_status",
            conversation_id="conversation_1",
            action=EventAction.QUEUE,
            payload_text="普通任务",
            origin=EventOrigin.FEISHU,
            received_at=self.received_at,
        )
        handling_at = self.received_at + timedelta(seconds=1)
        applied_at = self.received_at + timedelta(seconds=2)

        handling = transition_event_status(
            event,
            EventStatus.HANDLING,
            changed_at=handling_at,
        )
        applied = transition_event_status(
            handling,
            EventStatus.APPLIED,
            changed_at=applied_at,
            result_code="RUN_CREATED",
        )

        self.assertEqual(handling.status_changed_at, handling_at)
        self.assertEqual(applied.status_changed_at, applied_at)
        self.assertEqual(applied.result_code, "RUN_CREATED")

        with self.assertRaisesRegex(ValueError, "Illegal event transition"):
            transition_event_status(applied, EventStatus.HANDLING)

    def test_handling_event_can_be_requeued_after_process_recovery(self) -> None:
        event = create_agent_event(
            event_id="evt_recover",
            conversation_id="conversation_1",
            action=EventAction.QUEUE,
            payload_text="等待恢复的任务",
            origin=EventOrigin.FEISHU,
            received_at=self.received_at,
        )
        handling = transition_event_status(
            event,
            EventStatus.HANDLING,
            changed_at=self.received_at + timedelta(seconds=1),
        )
        pending = transition_event_status(
            handling,
            EventStatus.PENDING,
            changed_at=self.received_at + timedelta(seconds=2),
            result_code="RECOVERED_AFTER_RESTART",
        )

        self.assertEqual(pending.status, EventStatus.PENDING)
        self.assertEqual(pending.result_code, "RECOVERED_AFTER_RESTART")

    def test_run_supports_pause_and_automatic_resume(self) -> None:
        event = create_agent_event(
            event_id="evt_run",
            conversation_id="conversation_1",
            action=EventAction.QUEUE,
            payload_text="长任务",
            origin=EventOrigin.FEISHU,
            received_at=self.received_at,
        )
        run = create_event_run(event)
        running = transition_run_status(
            run,
            RunStatus.RUNNING,
            changed_at=self.received_at + timedelta(seconds=1),
        )
        paused = transition_run_status(
            running,
            RunStatus.PAUSED,
            changed_at=self.received_at + timedelta(seconds=2),
        )
        resumed = transition_run_status(
            paused,
            RunStatus.RUNNING,
            changed_at=self.received_at + timedelta(seconds=3),
        )

        encoded = json.dumps(resumed.to_dict())
        restored = EventRun.from_dict(json.loads(encoded))

        self.assertEqual(restored, resumed)
        self.assertEqual(resumed.status, RunStatus.RUNNING)

    def test_cancel_event_does_not_own_a_run(self) -> None:
        event = create_agent_event(
            event_id="evt_cancel",
            conversation_id="conversation_1",
            action=EventAction.CANCEL,
            payload_text="",
            target_event_id="evt_running",
            origin=EventOrigin.FEISHU,
            received_at=self.received_at,
        )

        with self.assertRaisesRegex(ValueError, "do not own an Agent run"):
            create_event_run(event)

    def test_cancel_event_requires_an_explicit_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "require target_event_id"):
            create_agent_event(
                event_id="evt_cancel_without_target",
                conversation_id="conversation_1",
                action=EventAction.CANCEL,
                payload_text="",
                origin=EventOrigin.FEISHU,
                received_at=self.received_at,
            )

    def test_naive_datetimes_and_backwards_status_time_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone information"):
            create_agent_event(
                event_id="evt_naive",
                conversation_id="conversation_1",
                action=EventAction.QUEUE,
                payload_text="普通任务",
                origin=EventOrigin.FEISHU,
                received_at=datetime(2026, 9, 1, 10, 0),
            )

        event = create_agent_event(
            event_id="evt_backwards",
            conversation_id="conversation_1",
            action=EventAction.QUEUE,
            payload_text="普通任务",
            origin=EventOrigin.FEISHU,
            received_at=self.received_at,
        )
        with self.assertRaisesRegex(ValueError, "cannot move backwards"):
            transition_event_status(
                event,
                EventStatus.HANDLING,
                changed_at=self.received_at - timedelta(seconds=1),
            )


if __name__ == "__main__":
    unittest.main()
